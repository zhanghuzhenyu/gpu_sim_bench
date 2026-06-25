import argparse
import json
import subprocess
import time
import gymnasium as gym
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS
sys.path.append(str(Path(__file__).resolve().parents[1]))
import roboverse_maniskill.franka_unified

def get_mem_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
    except Exception:
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
            return float(out.strip())
        except Exception:
            return 0.0
MEM_BASELINE = get_mem_mb()

def main():
    parser = argparse.ArgumentParser(description='ManiSkill unified Franka-only benchmark (A1 spec).')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=BENCHMARK_WARMUP_STEPS)
    parser.add_argument('--test_steps', type=int, default=BENCHMARK_TEST_STEPS)
    parser.add_argument('--mem_sample_interval', type=int, default=BENCHMARK_MEM_SAMPLE_INTERVAL)
    args = parser.parse_args()
    env = gym.make('FrankaUnified-v0', num_envs=args.num_envs, obs_mode='state', control_mode='pd_joint_pos', render_mode=None, sim_backend='gpu')
    env.reset(seed=0)
    mem_post_init = get_mem_mb()
    for _ in range(args.warmup_steps):
        action = env.action_space.sample()
        env.step(action)
    torch.cuda.synchronize()
    mem_post_warmup = get_mem_mb()
    peak_mem = mem_post_warmup
    start = time.perf_counter()
    for i in range(args.test_steps):
        action = env.action_space.sample()
        env.step(action)
        if args.mem_sample_interval > 0 and i % args.mem_sample_interval == 0:
            peak_mem = max(peak_mem, get_mem_mb())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mem_final = get_mem_mb()
    peak_mem = max(peak_mem, mem_final)
    fps = args.test_steps * args.num_envs / elapsed if elapsed > 0 else 0.0
    result = {'num_envs': args.num_envs, 'fps': float(fps), 'mem_baseline': float(MEM_BASELINE), 'mem_post_init': float(mem_post_init), 'mem_post_warmup': float(mem_post_warmup), 'mem_peak': float(peak_mem), 'mem_final': float(mem_final)}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}', flush=True)
    env.close()
if __name__ == '__main__':
    main()
