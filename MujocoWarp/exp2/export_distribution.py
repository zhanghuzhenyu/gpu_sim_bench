import os
import sys
import json
import time
import argparse
import numpy as np
import warp as wp
import mujoco
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mujoco_warp as mjw
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import EXP2_DEFAULT_NUM_ENVS, EXP2_DEFAULT_STEPS

def export_distribution(xml_path, num_envs, num_steps, output_json):
    if not os.path.exists(xml_path):
        print(f'Error: XML path {xml_path} not found.')
        return
    mjm = mujoco.MjModel.from_xml_path(xml_path)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    wp.init()
    device = 'cuda:0'
    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=4000, njmax=8000)
        with wp.ScopedCapture() as capture:
            mjw.step(m, d)
        graph = capture.graph
        print(f'Running simulation for {num_steps} steps...')
        for _ in range(num_steps):
            wp.capture_launch(graph)
        wp.synchronize()
        qpos = d.qpos.numpy()
        qvel = d.qvel.numpy()
        cube_bodies = []
        for i in range(mjm.nbody):
            name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith('cube'):
                jnt_id = -1
                for j in range(mjm.njnt):
                    if mjm.jnt_bodyid[j] == i:
                        jnt_id = j
                        break
                if jnt_id != -1:
                    cube_bodies.append({'id': len(cube_bodies), 'name': name, 'qpos_adr': mjm.jnt_qposadr[jnt_id], 'qvel_adr': mjm.jnt_dofadr[jnt_id]})
        serializable_data = {'metadata': {'simulator': 'MujocoWarp', 'num_envs': num_envs, 'dt': float(mjm.opt.timestep), 'substeps': int(mjm.opt.iterations), 'termination_step': num_steps, 'sample_time': num_steps * float(mjm.opt.timestep)}, 'environments': []}
        for env_idx in range(num_envs):
            cube_data = []
            positions = []
            for cube in cube_bodies:
                p_adr = cube['qpos_adr']
                v_adr = cube['qvel_adr']
                p = qpos[env_idx, p_adr:p_adr + 3]
                q = qpos[env_idx, p_adr + 3:p_adr + 7]
                lv = qvel[env_idx, v_adr:v_adr + 3]
                av = qvel[env_idx, v_adr + 3:v_adr + 6]
                cube_data.append({'name': cube['name'], 'cube_id': cube['id'], 'pos': p.tolist(), 'quat': q.tolist(), 'lin_vel': lv.tolist(), 'ang_vel': av.tolist()})
                positions.append(p)
            pos_np = np.array(positions)
            centroid = np.mean(pos_np, axis=0)
            max_spread = np.max(np.linalg.norm(pos_np - centroid, axis=1))
            serializable_data['environments'].append({'env_id': env_idx, 'num_cubes': len(cube_bodies), 'centroid': centroid.tolist(), 'max_spread': float(max_spread), 'cubes': cube_data})
        with open(output_json, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f'Results saved to {output_json}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_path', type=str)
    parser.add_argument('--num_envs', type=int, default=EXP2_DEFAULT_NUM_ENVS)
    parser.add_argument('--num_steps', type=int, default=EXP2_DEFAULT_STEPS)
    parser.add_argument('--output_json', type=str, required=True)
    args = parser.parse_args()
    export_distribution(args.xml_path, args.num_envs, args.num_steps, args.output_json)
