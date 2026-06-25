import argparse
import json
import time
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_mem_mb() -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / 1024 / 1024
except Exception:
    import subprocess

    def get_mem_mb() -> float:
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
            return float(result.strip())
        except Exception:
            return 0.0
import gymnasium as gym
import mani_skill.envs

def main():
    parser = argparse.ArgumentParser(description='ManiSkill Franka PickCube random-action benchmark.')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments.')
    args = parser.parse_args()
    mem_baseline = get_mem_mb()
    env = gym.make('PickCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos', render_mode=None, num_envs=args.num_envs, sim_backend='gpu', render_backend='none', enable_shadow=False, parallel_in_single_scene=False)
    np.random.seed(0)
    obs, _ = env.reset(seed=0, options=dict(reconfigure=True))
    mem_post_init = get_mem_mb()
    warmup_steps = BENCHMARK_WARMUP_STEPS
    for _ in range(warmup_steps):
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
    torch.cuda.synchronize()
    mem_post_warmup = get_mem_mb()
    test_steps = BENCHMARK_TEST_STEPS
    peak_mem = mem_post_warmup
    start_time = time.perf_counter()
    for i in range(test_steps):
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
        if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
            peak_mem = max(peak_mem, get_mem_mb())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    mem_final = get_mem_mb()
    peak_mem = max(peak_mem, mem_final)
    fps = test_steps * args.num_envs / elapsed if elapsed > 0 else 0.0
    result = {'num_envs': args.num_envs, 'fps': fps, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}')
    env.close()
if __name__ == '__main__':
    main()
