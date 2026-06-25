import argparse
import time
import json
import os
import mujoco
import warp as wp
import numpy as np
import mujoco_warp as mjw
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS, EXP1_CUBE_EDGE, EXP1_CUBE_MASS, EXP1_CUBE_SPACING, EXP1_DT, EXP1_INITIAL_CENTER_Z, EXP1_NUM_CUBES_PER_DIM

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

def _load_model(path):
    spec = mujoco.MjSpec.from_file(path)
    return spec.compile()


def _generate_scene_xml() -> str:
    m = EXP1_NUM_CUBES_PER_DIM
    spacing = EXP1_CUBE_SPACING
    half_n = (m - 1) / 2.0
    cube_half = EXP1_CUBE_EDGE / 2.0
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<mujoco model="falling_cubes">',
        f'  <option timestep="{EXP1_DT}" gravity="0 0 -9.81"/>',
        '  <worldbody>',
        '    <geom name="ground" type="plane" size="20 20 0.1" pos="0 0 0" friction="1 0.005 0.0001"/>',
    ]
    for i in range(m):
        for j in range(m):
            for k in range(m):
                px = (i - half_n) * spacing
                py = (j - half_n) * spacing
                pz = EXP1_INITIAL_CENTER_Z + (k - half_n) * spacing
                lines.extend([
                    f'    <body name="cube_{i}_{j}_{k}" pos="{px} {py} {pz}">',
                    '      <freejoint/>',
                    f'      <geom type="box" size="{cube_half} {cube_half} {cube_half}" mass="{EXP1_CUBE_MASS}"/>',
                    '    </body>',
                ])
    lines.extend(['  </worldbody>', '</mujoco>'])
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--nconmax', type=int, default=1500)
    parser.add_argument('--njmax', type=int, default=3000)
    args = parser.parse_args()
    mjm = mujoco.MjModel.from_xml_string(_generate_scene_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=args.num_envs, nconmax=args.nconmax, njmax=args.njmax)
        mem_post_init = get_mem()
        with wp.ScopedCapture() as capture:
            mjw.step(m, d)
        graph = capture.graph
        for _ in range(BENCHMARK_WARMUP_STEPS):
            wp.capture_launch(graph)
        wp.synchronize()
        mem_post_warmup = get_mem()
        test_steps = BENCHMARK_TEST_STEPS
        peak_mem = mem_post_warmup
        start_time = time.perf_counter()
        for i in range(test_steps):
            wp.capture_launch(graph)
            wp.synchronize()
            if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
                peak_mem = max(peak_mem, get_mem())
        total_time = time.perf_counter() - start_time
        mem_final = get_mem()
        peak_mem = max(peak_mem, mem_final)
        fps = test_steps * args.num_envs / total_time
        result = {'num_envs': args.num_envs, 'fps': fps, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
        print(f'BENCHMARK_RESULT:{json.dumps(result)}')
if __name__ == '__main__':
    main()
