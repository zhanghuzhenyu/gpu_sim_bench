import argparse
import json
import os
import subprocess
import time
import numpy as np
import torch
import sys
from pathlib import Path
import genesis as gs
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS, EXP3_DT, EXP3_ENV_SPACING

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
    parser.add_argument('--warmup_steps', type=int, default=BENCHMARK_WARMUP_STEPS)
    parser.add_argument('--test_steps', type=int, default=BENCHMARK_TEST_STEPS)
    parser.add_argument('--mem_sample_interval', type=int, default=BENCHMARK_MEM_SAMPLE_INTERVAL)
    args = parser.parse_args()
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(dt=EXP3_DT, gravity=(0, 0, -9.81)))
    mjcf_path = Path(__file__).resolve().with_name('scene_unified_single_franka_fixed.xml')
    if not mjcf_path.exists():
        raise FileNotFoundError(f'Cannot find unified Franka MJCF scene at {mjcf_path}')
    franka = scene.add_entity(gs.morphs.MJCF(file=str(mjcf_path)))
    scene.build(n_envs=args.num_envs, env_spacing=(EXP3_ENV_SPACING, EXP3_ENV_SPACING))
    mem_post_init = get_mem()
    joint_limits = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973], [0.0, 0.04]], dtype=np.float32)
    try:
        dof_count = franka.n_dofs
    except Exception:
        dof_count = None
    motor_dofs = np.arange(joint_limits.shape[0])

    def apply_position_targets(pos_targets: np.ndarray):
        if hasattr(franka, 'control_dofs_position'):
            franka.control_dofs_position(torch.tensor(pos_targets, device=gs.device), motor_dofs)
            return True
        try:
            franka.set_dofs_position(pos_targets)
            return True
        except Exception:
            try:
                franka.set_dofs_position(pos_targets, dofs_idx=list(motor_dofs))
                return True
            except Exception:
                return False
    for _ in range(args.warmup_steps):
        targets = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(args.num_envs, joint_limits.shape[0])).astype(np.float32)
        ok = apply_position_targets(targets)
        if not ok:
            raise RuntimeError('Failed to apply position targets in Genesis (no supported API)')
        scene.step()
    torch.cuda.synchronize()
    mem_post_warmup = get_mem()
    peak_mem = mem_post_warmup
    start_time = time.perf_counter()
    for i in range(args.test_steps):
        targets = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(args.num_envs, joint_limits.shape[0])).astype(np.float32)
        ok = apply_position_targets(targets)
        if not ok:
            raise RuntimeError('Failed to apply position targets in Genesis (no supported API)')
        scene.step()
        if args.mem_sample_interval > 0 and i % args.mem_sample_interval == 0:
            peak_mem = max(peak_mem, get_mem())
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    mem_final = get_mem()
    peak_mem = max(peak_mem, mem_final)
    fps = args.test_steps * args.num_envs / total_time
    result = {'num_envs': args.num_envs, 'fps': fps, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}')
if __name__ == '__main__':
    main()
