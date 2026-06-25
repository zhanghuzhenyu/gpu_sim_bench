import argparse
import time
import os
import math
import json
from datetime import datetime
import yaml
import numpy as np
import sys
from pathlib import Path
import genesis as gs
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import EXP2_DEFAULT_NUM_ENVS, EXP2_DEFAULT_STEPS, EXP2_DT

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def record_cube_positions(scene, n_envs, cube_start_idx, n_cubes, output_json, step, dt):
    print(f'\n[Info] Recording cube distributions at step {step}...')
    serializable_data = {'metadata': {'simulator': 'Genesis', 'num_envs': n_envs, 'dt': dt, 'substeps': 1, 'termination_step': step, 'sample_time': step * dt}, 'environments': []}
    for env_idx in range(n_envs):
        cube_data = []
        positions_array = []
        for cube_idx in range(n_cubes):
            entity = scene.entities[cube_start_idx + cube_idx]
            pos = entity.get_pos(envs_idx=env_idx).cpu().numpy().flatten()
            quat = entity.get_quat(envs_idx=env_idx).cpu().numpy().flatten()
            lin_vel = entity.get_vel(envs_idx=env_idx).cpu().numpy().flatten()
            ang_vel = entity.get_ang(envs_idx=env_idx).cpu().numpy().flatten()
            cube_data.append({'name': f'cube_{cube_idx}', 'cube_id': cube_idx, 'pos': pos.tolist(), 'quat': quat.tolist(), 'lin_vel': lin_vel.tolist(), 'ang_vel': ang_vel.tolist()})
            positions_array.append(pos)
        pos_np = np.array(positions_array)
        centroid = np.mean(pos_np, axis=0)
        max_spread = np.max(np.linalg.norm(pos_np - centroid, axis=1))
        serializable_data['environments'].append({'env_id': env_idx, 'num_cubes': n_cubes, 'centroid': centroid.tolist(), 'max_spread': float(max_spread), 'cubes': cube_data})
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f'[Info] Saved distribution to: {output_json}')

def main():
    parser = argparse.ArgumentParser(description='Genesis Slope Ball Collision Scene')
    parser.add_argument('-c', '--config', type=str, default='slope_cube_config.yaml', help='Path to config file, default: slope_cube_config.yaml')
    parser.add_argument('-n', '--n-envs', type=int, default=EXP2_DEFAULT_NUM_ENVS, help='Number of parallel environments.')
    parser.add_argument('-s', '--steps', type=int, default=EXP2_DEFAULT_STEPS, help='Number of simulation steps.')
    parser.add_argument('--no-viewer', action='store_true', default=False, help='Disable viewer')
    parser.add_argument('--record-step', type=int, default=None, help='Step number to record cube positions (None = no recording)')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the distribution JSON')
    args = parser.parse_args()
    config_path = args.config
    if not os.path.isabs(config_path) and (not os.path.exists(config_path)):
        config_path = os.path.join(os.path.dirname(__file__), os.path.basename(config_path))
    config = load_config(config_path)
    ground_plane_cfg = config['ground_plane']
    slope_cfg = config['slope']
    ball_cfg = config['ball']
    cube_cfg = config['cube']
    array_cfg = config['array']
    simulation_cfg = config.get('simulation', {})
    env_spacing = simulation_cfg.get('env_spacing', 2.0)
    print('=' * 60)
    print('Genesis Slope Ball Collision Scene Configuration')
    print('=' * 60)
    print(f"Ground plane:        {ground_plane_cfg['length']}m x {ground_plane_cfg['width']}m x {ground_plane_cfg['thickness']}m")
    print(f"Ground center:       ({ground_plane_cfg['center_x']}, {ground_plane_cfg['center_y']}, {ground_plane_cfg['center_z']})m")
    print(f"Slope angle:         {slope_cfg['angle_deg']} deg")
    print(f"Slope dimensions:    {slope_cfg['length']}m x {slope_cfg['width']}m x {slope_cfg['thickness']}m")
    print(f"Slope center:        ({slope_cfg['center_x']}, {slope_cfg['center_y']}, {slope_cfg['center_z']})m")
    print(f"Slope friction:      mu={slope_cfg['static_friction']}")
    print(f"Slope restitution:   e={slope_cfg['restitution']}")
    print(f"Ball radius:         {ball_cfg['radius']}m")
    print(f"Ball mass:           {ball_cfg['mass']}kg")
    print(f"Ball center:         ({ball_cfg['center_x']}, {ball_cfg['center_y']}, {ball_cfg['center_z']})m")
    print(f"Ball friction:       mu={ball_cfg['static_friction']}")
    print(f"Ball restitution:    e={ball_cfg['restitution']}")
    print(f"Cube array size:     {array_cfg['m']} x {array_cfg['m']} x {array_cfg['m']}")
    print(f"Cube size:           {cube_cfg['size']}m")
    print(f"Cube mass:           {cube_cfg['mass']}kg")
    print(f"Array center:        ({array_cfg['center_x']}, {array_cfg['center_y']}, {array_cfg['center_z']})m")
    print(f"Cube friction:       mu={cube_cfg['static_friction']}")
    print(f'Parallel envs:       {args.n_envs}')
    print(f'Env spacing:         {env_spacing}m')
    print(f'Simulation steps:    {args.steps}')
    print('=' * 60)
    gs.init(backend=gs.gpu)
    m = array_cfg['m']
    n_cubes = m ** 3
    max_collision_pairs = max(4096, n_cubes * (n_cubes + 1) // 2 + n_cubes * 2 + 100)
    scene = gs.Scene(show_viewer=not args.no_viewer, rigid_options=gs.options.RigidOptions(dt=EXP2_DT, gravity=(0, 0, -9.81), max_collision_pairs=max_collision_pairs), viewer_options=gs.options.ViewerOptions(camera_pos=(5.0, -8.0, 4.0), camera_lookat=(0.0, 0.0, 1.0), camera_fov=50), profiling_options=gs.options.ProfilingOptions(show_FPS=True, FPS_tracker_alpha=0.95))
    ground_plane = scene.add_entity(morph=gs.morphs.Box(pos=(ground_plane_cfg['center_x'], ground_plane_cfg['center_y'], ground_plane_cfg['center_z']), size=(ground_plane_cfg['length'], ground_plane_cfg['width'], ground_plane_cfg['thickness']), fixed=True), material=gs.materials.Rigid(rho=600.0, friction=ground_plane_cfg['static_friction'], coup_restitution=ground_plane_cfg['restitution']), surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0)))
    angle_rad = math.radians(slope_cfg['angle_deg'])
    slope = scene.add_entity(morph=gs.morphs.Box(pos=(slope_cfg['center_x'], slope_cfg['center_y'], slope_cfg['center_z']), size=(slope_cfg['length'], slope_cfg['width'], slope_cfg['thickness']), euler=(0, slope_cfg['angle_deg'], 0), fixed=True), material=gs.materials.Rigid(rho=600.0, friction=slope_cfg['static_friction'], coup_restitution=slope_cfg['restitution']), surface=gs.surfaces.Default(color=(0.7, 0.5, 0.3, 1.0)))
    ball_volume = 4 / 3 * math.pi * ball_cfg['radius'] ** 3
    ball_density = ball_cfg['mass'] / ball_volume
    ball = scene.add_entity(morph=gs.morphs.Sphere(pos=(ball_cfg['center_x'], ball_cfg['center_y'], ball_cfg['center_z']), radius=ball_cfg['radius']), material=gs.materials.Rigid(rho=ball_density, friction=ball_cfg['static_friction'], coup_restitution=ball_cfg['restitution']), surface=gs.surfaces.Default(color=(0.8, 0.8, 0.9, 1.0)))
    m = array_cfg['m']
    spacing = array_cfg['spacing']
    cube_size = cube_cfg['size']
    cube_volume = cube_size ** 3
    cube_density = cube_cfg['mass'] / cube_volume
    start_x = array_cfg['center_x'] - (m - 1) * spacing / 2
    start_y = array_cfg['center_y'] - (m - 1) * spacing / 2
    start_z = array_cfg['center_z'] - (m - 1) * spacing / 2
    n_cubes = m ** 3
    print(f'\n[Info] Creating {m}x{m}x{m} = {n_cubes} cubes...')
    print(f"[Info] Array center: ({array_cfg['center_x']}, {array_cfg['center_y']}, {array_cfg['center_z']})m")
    print(f'[Info] Array start:  ({start_x}, {start_y}, {start_z})m')
    cube_start_idx = 3
    for i in range(m):
        for j in range(m):
            for k in range(m):
                cube_x = start_x + i * spacing
                cube_y = start_y + j * spacing
                cube_z = start_z + k * spacing
                scene.add_entity(morph=gs.morphs.Box(pos=(cube_x, cube_y, cube_z), size=(cube_size, cube_size, cube_size)), material=gs.materials.Rigid(rho=cube_density, friction=cube_cfg['static_friction'], coup_restitution=cube_cfg['restitution']), surface=gs.surfaces.Default(color=(0.9, 0.7, 0.5, 1.0)))
    print(f'\n[Info] Building scene with {args.n_envs} environment(s)...')
    scene.build(n_envs=args.n_envs, env_spacing=(env_spacing, env_spacing))
    print('[Info] Scene built successfully!')
    print(f'\n[Info] Running {args.steps} simulation steps...')
    print(f'[Info] Expected simulation time: {args.steps * EXP2_DT:.2f}s')
    if args.record_step is not None:
        print(f'[Info] Will record cube positions at step {args.record_step}')
    start_time = time.time()
    for step in range(args.steps):
        scene.step()
        if args.record_step is not None and step == args.record_step:
            record_cube_positions(scene, args.n_envs, cube_start_idx, n_cubes, args.output_json, step, EXP2_DT)
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps = (step + 1) / elapsed
            print(f'[Progress] Step {step + 1}/{args.steps}, FPS: {fps:.1f}')
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = args.steps / total_time
    print('\n' + '=' * 60)
    print('Simulation Complete')
    print('=' * 60)
    print(f'Total steps:         {args.steps}')
    print(f'Total time:          {total_time:.2f}s')
    print(f'Average FPS:         {avg_fps:.1f}')
    print(f'Step time:           {total_time / args.steps * 1000:.2f}ms')
    print('=' * 60)
if __name__ == '__main__':
    main()
