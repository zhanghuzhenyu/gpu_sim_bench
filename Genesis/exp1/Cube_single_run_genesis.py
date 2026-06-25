import argparse
import time
import torch
import numpy as np
import json
import subprocess
import os
import sys
from pathlib import Path
import genesis as gs
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS, EXP1_CUBE_EDGE, EXP1_CUBE_SPACING, EXP1_DT, EXP1_ENV_SPACING, EXP1_INITIAL_CENTER_Z, EXP1_NUM_CUBES_PER_DIM

def get_mem():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
    except Exception:
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
            return float(result.strip())
        except Exception:
            return 0.0
mem_baseline = get_mem()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=64)
    args = parser.parse_args()
    gs.init(backend=gs.gpu)
    m, spacing, center_z = (EXP1_NUM_CUBES_PER_DIM, EXP1_CUBE_SPACING, EXP1_INITIAL_CENTER_Z)
    half_n = (m - 1) / 2.0
    scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(dt=EXP1_DT, gravity=(0, 0, -9.81)))
    for i in range(m):
        for j in range(m):
            for k in range(m):
                px = (i - half_n) * spacing
                py = (j - half_n) * spacing
                pz = center_z + (k - half_n) * spacing
                scene.add_entity(morph=gs.morphs.Box(size=(EXP1_CUBE_EDGE, EXP1_CUBE_EDGE, EXP1_CUBE_EDGE), pos=(px, py, pz)))
    scene.add_entity(gs.morphs.Plane())
    scene.build(n_envs=args.num_envs, env_spacing=(EXP1_ENV_SPACING, EXP1_ENV_SPACING))
    mem_post_init = get_mem()
    for _ in range(BENCHMARK_WARMUP_STEPS):
        scene.step()
    torch.cuda.synchronize()
    mem_post_warmup = get_mem()
    test_steps = BENCHMARK_TEST_STEPS
    peak_mem = mem_post_warmup
    start_time = time.perf_counter()
    for i in range(test_steps):
        scene.step()
        if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
            peak_mem = max(peak_mem, get_mem())
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    mem_final = get_mem()
    peak_mem = max(peak_mem, mem_final)
    result = {'num_envs': args.num_envs, 'fps': test_steps * args.num_envs / total_time, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}')
if __name__ == '__main__':
    main()
