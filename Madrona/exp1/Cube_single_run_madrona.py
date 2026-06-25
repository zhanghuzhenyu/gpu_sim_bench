import argparse
import time
import json
import os
import falling_cubes
import pynvml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS
pynvml.nvmlInit()
_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def mem_used_mb():
    return pynvml.nvmlDeviceGetMemoryInfo(_handle).used / 1024 / 1024
MEM_BASELINE = mem_used_mb()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, required=True, help='Madrona num_worlds')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    exec_mode = falling_cubes.madrona.ExecMode.CUDA
    sim = falling_cubes.SimManager(exec_mode=exec_mode, gpu_id=args.gpu_id, num_worlds=args.num_envs, auto_reset=True, rand_seed=5, enable_batch_renderer=False)
    mem_post_init = mem_used_mb()
    for _ in range(BENCHMARK_WARMUP_STEPS):
        sim.step()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize(args.gpu_id)
    except Exception:
        pass
    mem_post_warmup = mem_used_mb()
    test_steps = BENCHMARK_TEST_STEPS
    peak = mem_post_warmup
    start = time.perf_counter()
    for i in range(test_steps):
        sim.step()
        if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
            peak = max(peak, mem_used_mb())
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize(args.gpu_id)
    except Exception:
        pass
    elapsed = time.perf_counter() - start
    mem_final = mem_used_mb()
    peak = max(peak, mem_final)
    fps = test_steps * args.num_envs / elapsed
    out = {'num_envs': args.num_envs, 'fps': fps, 'mem_baseline': MEM_BASELINE, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak, 'mem_final': mem_final}
    print('BENCHMARK_RESULT:' + json.dumps(out))
if __name__ == '__main__':
    main()
