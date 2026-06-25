import argparse
import os
import math
import yaml
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import jax
from jax import numpy as jp
from mujoco import mjx
import mujoco
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import EXP2_DEFAULT_NUM_ENVS, EXP2_DEFAULT_STEPS

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_mjcf(config):
    ground_plane_params = config.get('ground_plane', {})
    slope_params = config['slope']
    ball_params = config['ball']
    cube_params = config['cube']
    array_params = config['array']
    slope_angle_rad = math.radians(slope_params['angle_deg'])
    xml_lines = ['<?xml version="1.0" encoding="utf-8"?>', '<mujoco model="slope_cube_mjx">', '  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>', '  <option timestep="0.01" iterations="50" solver="Newton" tolerance="1e-10">', '    <flag eulerdamp="disable"/>', '  </option>', '  <asset>', '    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>', '    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>', '    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true" reflectance=".2"/>', '  </asset>', '  <worldbody>', '    <geom name="ground_infinite" type="plane" size="20 20 0.1" pos="0 0 0" rgba="0.5 0.5 0.5 0.3" friction="0.5 0.35 0.0001"/>']
    if ground_plane_params:
        gp = ground_plane_params
        xml_lines.append(f'''    <body name="ground_plane" pos="{gp['center_x']} {gp['center_y']} {gp['center_z']}">''')
        xml_lines.append(f'''      <geom name="ground_plane_geom" type="box" size="{gp['length'] / 2} {gp['width'] / 2} {gp['thickness'] / 2}" rgba="0.7 0.5 0.3 1" friction="{gp['static_friction']} {gp['dynamic_friction']} 0.0001"/>''')
        xml_lines.append('    </body>')
    slope_quat = f'{math.cos(slope_angle_rad / 2):.7f} 0 {math.sin(slope_angle_rad / 2):.7f} 0'
    xml_lines.append(f'''    <body name="slope" pos="{slope_params['center_x']} {slope_params['center_y']} {slope_params['center_z']}" quat="{slope_quat}">''')
    xml_lines.append(f'''      <geom name="slope_geom" type="box" size="{slope_params['length'] / 2} {slope_params['width'] / 2} {slope_params['thickness'] / 2}" rgba="0.6 0.4 0.2 1" friction="{slope_params['static_friction']} {slope_params['dynamic_friction']} 0.0001"/>''')
    xml_lines.append('    </body>')
    xml_lines.append(f'''    <body name="ball" pos="{ball_params['center_x']} {ball_params['center_y']} {ball_params['center_z']}">''')
    xml_lines.append(f'''      <geom name="ball_geom" type="sphere" size="{ball_params['radius']}" rgba="0.8 0.2 0.2 1" friction="{ball_params['static_friction']} {ball_params['dynamic_friction']} 0.0001" mass="{ball_params['mass']}"/>''')
    xml_lines.append('      <freejoint name="ball_free_joint"/>')
    xml_lines.append('    </body>')
    m = array_params['m']
    spacing = array_params['spacing']
    half_size = (m - 1) * spacing / 2
    start_x, start_y, start_z = (array_params['center_x'] - half_size, array_params['center_y'] - half_size, array_params['center_z'] - half_size)
    cube_idx = 0
    for k in range(m):
        for i in range(m):
            for j in range(m):
                cx, cy, cz = (start_x + i * spacing, start_y + j * spacing, start_z + k * spacing)
                xml_lines.append(f'    <body name="cube{cube_idx}" pos="{cx} {cy} {cz}">')
                xml_lines.append(f'''      <geom name="cube_geom{cube_idx}" type="box" size="{cube_params['size'] / 2} {cube_params['size'] / 2} {cube_params['size'] / 2}" rgba="0.4 0.6 0.8 1" friction="{cube_params['static_friction']} {cube_params['dynamic_friction']} 0.0001" mass="{cube_params['mass']}"/>''')
                xml_lines.append(f'      <freejoint name="cube_free_joint{cube_idx}"/>')
                xml_lines.append('    </body>')
                cube_idx += 1
    xml_lines.extend(['  </worldbody>', '</mujoco>'])
    return '\n'.join(xml_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=str(Path(__file__).resolve().parents[2] / 'Genesis' / 'exp2' / 'slope_cube_config.yaml'))
    parser.add_argument('-n', '--n-envs', type=int, default=EXP2_DEFAULT_NUM_ENVS)
    parser.add_argument('-s', '--steps', type=int, default=EXP2_DEFAULT_STEPS)
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the distribution JSON')
    args = parser.parse_args()
    config = load_config(args.config)
    mjcf_string = generate_mjcf(config)
    model = mujoco.MjModel.from_xml_string(mjcf_string)
    mjx_model = mjx.put_model(model)

    @jax.vmap
    def step(data):
        return mjx.step(mjx_model, data)
    data = mjx.make_data(mjx_model)
    batch_data = jax.tree_util.tree_map(lambda x: jp.array(np.tile(x, (args.n_envs,) + (1,) * x.ndim)), data)
    jit_step = jax.jit(step)
    print(f'Starting simulation for {args.n_envs} envs, {args.steps} steps...')
    start_time = time.time()
    for i in range(args.steps):
        batch_data = jit_step(batch_data)
        if (i + 1) % 500 == 0:
            print(f'Step {i + 1}/{args.steps}')
    jax.block_until_ready(batch_data)
    end_time = time.time()
    m = config['array']['m']
    n_cubes = m ** 3
    cube_start_body_id = 4
    dt = model.opt.timestep
    serializable_data = {'metadata': {'simulator': 'MJX', 'num_envs': args.n_envs, 'dt': float(dt), 'substeps': int(model.opt.iterations), 'termination_step': args.steps, 'sample_time': float(args.steps * dt)}, 'environments': []}
    for env_idx in range(args.n_envs):
        cube_data = []
        positions_array = []
        for c_idx in range(n_cubes):
            body_id = cube_start_body_id + c_idx
            pos = batch_data.xpos[env_idx, body_id]
            quat = batch_data.xquat[env_idx, body_id]
            cube_info = {'name': f'cube_{c_idx}', 'cube_id': c_idx, 'pos': [float(pos[0]), float(pos[1]), float(pos[2])], 'quat': [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]}
            if hasattr(batch_data, 'cvel'):
                vel = batch_data.cvel[env_idx, body_id]
                cube_info['lin_vel'] = [float(vel[3]), float(vel[4]), float(vel[5])]
                cube_info['ang_vel'] = [float(vel[0]), float(vel[1]), float(vel[2])]
            cube_data.append(cube_info)
            positions_array.append(pos)
        pos_np = np.array(positions_array)
        centroid = np.mean(pos_np, axis=0)
        max_spread = np.max(np.linalg.norm(pos_np - centroid, axis=1))
        serializable_data['environments'].append({'env_id': env_idx, 'num_cubes': n_cubes, 'centroid': centroid.tolist(), 'max_spread': float(max_spread), 'cubes': cube_data})
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2)
        print(f'Results saved to {args.output_json}')
    else:
        os.makedirs('cube_positions', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'cube_positions/mjx_results_{ts}.json', 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f'Results saved to cube_positions/mjx_results_{ts}.json')
if __name__ == '__main__':
    main()
