#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
import jax
import jax.numpy as jp
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import EXP2_DEFAULT_NUM_ENVS, EXP2_DEFAULT_STEPS
import slope_cube_env

def export_distribution(num_envs: int, num_steps: int, output_json: str):
    env = slope_cube_env.load()
    dt = float(env._ctrl_dt)
    substeps = int(env._ctrl_dt / env._sim_dt)
    print(f'[MujocoPlayground] Initializing envs: num_envs={num_envs}, dt={dt}, substeps={substeps}')
    rng = jax.random.PRNGKey(0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(rng, num_envs)
    state = reset_fn(keys)
    actions = jp.zeros((num_envs, env.action_size))
    print(f'[MujocoPlayground] Running simulation for {num_steps} steps...')
    t0 = time.perf_counter()
    for _ in range(num_steps):
        state = step_fn(state, actions)
    state.data.qpos.block_until_ready()
    elapsed = time.perf_counter() - t0
    print(f'[MujocoPlayground] Done stepping in {elapsed:.3f}s')
    qpos = jax.device_get(state.data.qpos)
    qvel = jax.device_get(state.data.qvel)
    num_cubes = env.num_cubes
    serializable_data = {'metadata': {'simulator': 'MujocoPlayground', 'num_envs': int(num_envs), 'dt': float(dt), 'substeps': int(substeps), 'termination_step': int(num_steps), 'sample_time': float(num_steps * dt)}, 'environments': []}
    for env_idx in range(num_envs):
        cube_data = []
        positions = []
        ball_qpos_n = 7
        ball_qvel_n = 6
        for cube_id in range(num_cubes):
            p_adr = ball_qpos_n + cube_id * 7
            v_adr = ball_qvel_n + cube_id * 6
            p = qpos[env_idx, p_adr:p_adr + 3]
            q = qpos[env_idx, p_adr + 3:p_adr + 7]
            lv = qvel[env_idx, v_adr:v_adr + 3]
            av = qvel[env_idx, v_adr + 3:v_adr + 6]
            cube_data.append({'name': f'cube_{cube_id}', 'cube_id': int(cube_id), 'pos': [float(x) for x in p], 'quat': [float(x) for x in q], 'lin_vel': [float(x) for x in lv], 'ang_vel': [float(x) for x in av]})
            positions.append(p)
        import numpy as np
        pos_np = np.array(positions)
        centroid = np.mean(pos_np, axis=0)
        max_spread = np.max(np.linalg.norm(pos_np - centroid, axis=1))
        serializable_data['environments'].append({'env_id': int(env_idx), 'num_cubes': int(num_cubes), 'centroid': [float(x) for x in centroid], 'max_spread': float(max_spread), 'cubes': cube_data})
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f'[MujocoPlayground] Results saved to {out_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=EXP2_DEFAULT_NUM_ENVS)
    parser.add_argument('--num_steps', type=int, default=EXP2_DEFAULT_STEPS)
    parser.add_argument('--output_json', type=str, required=True)
    args = parser.parse_args()
    export_distribution(args.num_envs, args.num_steps, args.output_json)
if __name__ == '__main__':
    main()
