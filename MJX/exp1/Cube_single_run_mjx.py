import argparse
import time
import json
import os
import sys
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS, EXP1_CUBE_EDGE, EXP1_CUBE_MASS, EXP1_CUBE_SPACING, EXP1_INITIAL_CENTER_Z, EXP1_NUM_CUBES_PER_DIM


def generate_cube_array_xml() -> str:
    m = EXP1_NUM_CUBES_PER_DIM
    spacing = EXP1_CUBE_SPACING
    half_n = (m - 1) / 2.0
    cube_half = EXP1_CUBE_EDGE / 2.0
    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<mujoco model="cube_array_mjx">',
        '  <option timestep="0.01" gravity="0 0 -9.81" iterations="20" solver="Newton"/>',
        '  <worldbody>',
        '    <geom name="ground" type="plane" size="20 20 0.1" pos="0 0 0" friction="1 0.005 0.0001"/>',
    ]
    for i in range(m):
        for j in range(m):
            for k in range(m):
                px = (i - half_n) * spacing
                py = (j - half_n) * spacing
                pz = EXP1_INITIAL_CENTER_Z + (k - half_n) * spacing
                xml_lines.extend([
                    f'    <body name="cube_{i}_{j}_{k}" pos="{px} {py} {pz}">',
                    '      <freejoint/>',
                    f'      <geom type="box" size="{cube_half} {cube_half} {cube_half}" mass="{EXP1_CUBE_MASS}" friction="1 0.005 0.0001"/>',
                    '    </body>',
                ])
    xml_lines.extend(['  </worldbody>', '</mujoco>'])
    return '\n'.join(xml_lines)

def get_mem():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
        return float(result.strip())
    except Exception:
        return 0.0
mem_baseline = get_mem()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    xml = generate_cube_array_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.gravity = [0, 0, -9.81]
    mjx_model = mjx.put_model(model)
    rng = jax.random.PRNGKey(args.seed)

    @jax.vmap
    def make_data(_):
        return mjx.make_data(mjx_model)
    mjx_data = make_data(jax.random.split(rng, args.num_envs))
    mem_post_init = get_mem()

    @jax.jit
    def batched_step(data):
        return jax.vmap(lambda d: mjx.step(mjx_model, d))(data)
    mjx_data = batched_step(mjx_data)
    jax.block_until_ready(mjx_data.qpos)
    for _ in range(BENCHMARK_WARMUP_STEPS - 1):
        mjx_data = batched_step(mjx_data)
    jax.block_until_ready(mjx_data.qpos)
    mem_post_warmup = get_mem()
    test_steps = BENCHMARK_TEST_STEPS
    peak_mem = mem_post_warmup
    start_time = time.perf_counter()
    for i in range(test_steps):
        mjx_data = batched_step(mjx_data)
        if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
            jax.block_until_ready(mjx_data.qpos)
            peak_mem = max(peak_mem, get_mem())
    jax.block_until_ready(mjx_data.qpos)
    total_time = time.perf_counter() - start_time
    mem_final = get_mem()
    peak_mem = max(peak_mem, mem_final)
    fps = test_steps * args.num_envs / total_time
    result = {'num_envs': args.num_envs, 'fps': fps, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}')
if __name__ == '__main__':
    main()
