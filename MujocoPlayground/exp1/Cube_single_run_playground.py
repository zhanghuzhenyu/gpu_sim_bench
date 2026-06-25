#!/usr/bin/env python3
import argparse
import time
import json
import os
import sys
import jax
import jax.numpy as jp
import pynvml
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS, EXP1_CUBE_HALF_EXTENT, EXP1_CUBE_SPACING, EXP1_INITIAL_CENTER_Z, EXP1_NUM_CUBES_PER_DIM
import cube_array_env
pynvml.nvmlInit()
_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def mem_used_mb() -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(_handle).used / 1024 / 1024
MEM_BASELINE = mem_used_mb()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    num_envs = args.num_envs
    env_config = {'num_cubes_per_dim': EXP1_NUM_CUBES_PER_DIM, 'cube_size': EXP1_CUBE_HALF_EXTENT, 'cube_spacing': EXP1_CUBE_SPACING, 'initial_height': EXP1_INITIAL_CENTER_Z, 'episode_length': BENCHMARK_WARMUP_STEPS + BENCHMARK_TEST_STEPS}
    env = cube_array_env.load(config_overrides=env_config)
    mem_post_init = mem_used_mb()
    rng = jax.random.PRNGKey(args.seed)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(rng, num_envs)
    state = reset_fn(keys)
    actions = jp.zeros((num_envs, env.action_size))
    state = step_fn(state, actions)
    state.data.qpos.block_until_ready()
    for _ in range(BENCHMARK_WARMUP_STEPS):
        state = step_fn(state, actions)
    state.data.qpos.block_until_ready()
    mem_post_warmup = mem_used_mb()
    peak = mem_post_warmup
    test_steps = BENCHMARK_TEST_STEPS
    start = time.perf_counter()
    for i in range(test_steps):
        state = step_fn(state, actions)
    state.data.qpos.block_until_ready()
    elapsed = time.perf_counter() - start
    peak = max(peak, mem_used_mb())
    mem_final = mem_used_mb()
    peak = max(peak, mem_final)
    fps = test_steps * num_envs / elapsed
    out = {'num_envs': num_envs, 'fps': float(fps), 'mem_baseline': float(MEM_BASELINE), 'mem_post_init': float(mem_post_init), 'mem_post_warmup': float(mem_post_warmup), 'mem_peak': float(peak), 'mem_final': float(mem_final)}
    print('BENCHMARK_RESULT:' + json.dumps(out))
if __name__ == '__main__':
    main()
