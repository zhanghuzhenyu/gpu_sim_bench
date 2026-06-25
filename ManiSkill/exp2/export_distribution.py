import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import EXP2_DEFAULT_NUM_ENVS, EXP2_DEFAULT_STEPS, EXP2_DT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mani_skill.utils.structs.pose import Pose
from benchmark_zhang_slope.slope_ball_impact_visualize import SlopeBallImpactEnv

def export_distribution(num_envs, num_steps, config_path, output_json):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    sim_freq = int(1.0 / EXP2_DT)
    control_freq = sim_freq
    dt = EXP2_DT
    print(f'Initializing ManiSkill Env (Headless)...')
    env = SlopeBallImpactEnv(num_envs=num_envs, obs_mode='state', render_mode=None, control_mode='none', config_path=config_path, sim_config={'sim_freq': sim_freq, 'control_freq': control_freq})
    env.reset(seed=42)
    print(f'Running simulation for {num_steps} steps...')
    substeps = sim_freq // control_freq
    control_steps = num_steps // substeps
    for _ in range(control_steps):
        env.step(None)
    serializable_data = {'metadata': {'simulator': 'ManiSkill', 'num_envs': num_envs, 'dt': float(dt), 'substeps': substeps, 'termination_step': num_steps, 'sample_time': num_steps * dt}, 'environments': []}
    for env_idx in range(num_envs):
        cube_data = []
        positions = []
        for i, cube in enumerate(env.cubes):
            p = cube.pose.p[env_idx].cpu().numpy()
            q = cube.pose.q[env_idx].cpu().numpy()
            lv = cube.linear_velocity[env_idx].cpu().numpy()
            av = cube.angular_velocity[env_idx].cpu().numpy()
            cube_data.append({'name': f'cube_{i}', 'cube_id': i, 'pos': p.tolist(), 'quat': q.tolist(), 'lin_vel': lv.tolist(), 'ang_vel': av.tolist()})
            positions.append(p)
        pos_np = np.array(positions)
        centroid = np.mean(pos_np, axis=0)
        max_spread = np.max(np.linalg.norm(pos_np - centroid, axis=1))
        serializable_data['environments'].append({'env_id': env_idx, 'num_cubes': len(env.cubes), 'centroid': centroid.tolist(), 'max_spread': float(max_spread), 'cubes': cube_data})
    with open(output_json, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f'Results saved to {output_json}')
    env.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=EXP2_DEFAULT_NUM_ENVS)
    parser.add_argument('--num_steps', type=int, default=EXP2_DEFAULT_STEPS)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    args = parser.parse_args()
    export_distribution(args.num_envs, args.num_steps, args.config, args.output_json)
