#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import EXP2_DEFAULT_STEPS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
try:
    import slope_scene
    from slope_scene import madrona
except ImportError as e:
    print(f'Error importing slope_scene module: {e}')
    sys.exit(1)

def export_distribution(num_worlds, num_steps, gpu_id=0, output_json='madrona_dist.json'):
    print(f'Initializing Madrona (CUDA Mode)...')
    exec_mode_enum = slope_scene.madrona.ExecMode.CUDA
    sim = slope_scene.SimManager(exec_mode_enum, gpu_id, num_worlds, 42, False, False)
    print(f'Running simulation for {num_steps} steps...')
    for step in range(num_steps):
        sim.step()
    print('Retrieving tensors...')
    pos_tensor = sim.cube_positions_tensor().to_torch().cpu().numpy()
    if hasattr(sim, 'cube_rotations_tensor'):
        quat_tensor = sim.cube_rotations_tensor().to_torch().cpu().numpy()
    else:
        print('Warning: cube_rotations_tensor not found in bindings. Exporting identity quaternions.')
        quat_tensor = np.zeros((num_worlds, pos_tensor.shape[1], 4), dtype=np.float32)
        quat_tensor[..., 0] = 1.0
    if hasattr(sim, 'cube_velocities_tensor'):
        vel_tensor = sim.cube_velocities_tensor().to_torch().cpu().numpy()
    else:
        print('Warning: cube_velocities_tensor not found in bindings. Exporting zeros for velocities.')
        vel_tensor = np.zeros((num_worlds, pos_tensor.shape[1], 6), dtype=np.float32)
    cube_start_idx = 3
    num_cubes = 27
    serializable_data = {'metadata': {'simulator': 'Madrona', 'num_envs': num_worlds, 'dt': 0.01, 'substeps': 4, 'termination_step': num_steps, 'sample_time': num_steps * 0.01}, 'environments': []}
    for world_idx in range(num_worlds):
        cube_data = []
        positions_array = []
        for c_idx in range(num_cubes):
            entity_idx = cube_start_idx + c_idx
            p = pos_tensor[world_idx, entity_idx]
            q = quat_tensor[world_idx, entity_idx]
            v = vel_tensor[world_idx, entity_idx]
            cube_data.append({'name': f'cube_{c_idx}', 'cube_id': c_idx, 'pos': [float(p[0]), float(p[1]), float(p[2])], 'quat': [float(q[0]), float(q[1]), float(q[2]), float(q[3])], 'lin_vel': [float(v[3]), float(v[4]), float(v[5])], 'ang_vel': [float(v[0]), float(v[1]), float(v[2])]})
            positions_array.append(p)
        pos_np = np.array(positions_array)
        centroid = np.mean(pos_np, axis=0)
        max_spread = np.max(np.linalg.norm(pos_np - centroid, axis=1))
        serializable_data['environments'].append({'env_id': world_idx, 'num_cubes': num_cubes, 'centroid': centroid.tolist(), 'max_spread': float(max_spread), 'cubes': cube_data})
    with open(output_json, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f'Results saved to {output_json}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-worlds', type=int, default=16)
    parser.add_argument('--num-steps', type=int, default=EXP2_DEFAULT_STEPS)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--output-json', type=str, required=True)
    args = parser.parse_args()
    export_distribution(args.num_worlds, args.num_steps, args.gpu_id, args.output_json)
