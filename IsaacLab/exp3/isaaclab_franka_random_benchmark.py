import argparse
import json
import time
import os
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

    def get_mem_mb() -> float:
        return 0.0
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description='Franka random-action benchmark in Isaac Lab.')
parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments.')
parser.add_argument('--task', type=str, default='Isaac-Lift-Cube-Franka-v0', help='Isaac Lab task name to benchmark.')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
mem_baseline = get_mem_mb()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs_space = env.observation_space
    act_space = env.action_space
    print(f'[INFO] Obs space: {obs_space}')
    print(f'[INFO] Act space: {act_space}')
    env.reset()
    mem_post_init = get_mem_mb()
    warmup_steps = BENCHMARK_WARMUP_STEPS
    for _ in range(warmup_steps):
        with torch.inference_mode():
            actions = 2 * torch.rand(act_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)
    torch.cuda.synchronize()
    mem_post_warmup = get_mem_mb()
    test_steps = BENCHMARK_TEST_STEPS
    peak_mem = mem_post_warmup
    start_time = time.perf_counter()
    for i in range(test_steps):
        with torch.inference_mode():
            actions = 2 * torch.rand(act_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)
        if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
            peak_mem = max(peak_mem, get_mem_mb())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    mem_final = get_mem_mb()
    peak_mem = max(peak_mem, mem_final)
    fps = test_steps * args_cli.num_envs / elapsed if elapsed > 0 else 0.0
    result = {'num_envs': args_cli.num_envs, 'fps': fps, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}')
    env.close()
if __name__ == '__main__':
    try:
        main()
    finally:
        simulation_app.close()
