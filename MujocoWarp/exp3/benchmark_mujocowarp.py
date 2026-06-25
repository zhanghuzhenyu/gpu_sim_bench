import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS

class BenchmarkTracker:

    def __init__(self, gpu_id: int=0):
        self.gpu_id = gpu_id
        self.nvml_initialized = False
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.nvml_initialized = True
        except Exception as e:
            print(f'[Warning] NVML Init failed (will fallback to nvidia-smi): {e}')

    def get_mem_used_mb(self) -> float:
        if self.nvml_initialized and self.handle is not None:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return info.used / 1024 / 1024
        try:
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
            return float(result.strip())
        except Exception:
            return 0.0

def _load_franka_scene_xml() -> Path:
    base = Path(__file__).resolve().parent.parent / 'benchmark' / 'franka_emika_panda'
    p1 = base / 'scene_with_objects.xml'
    if p1.exists():
        return p1
    p2 = base / 'scene.xml'
    if p2.exists():
        return p2
    raise FileNotFoundError(f'Cannot find Franka scene.xml under {base}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--nconmax', type=int, default=16)
    parser.add_argument('--njmax', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=BENCHMARK_WARMUP_STEPS)
    parser.add_argument('--test_steps', type=int, default=BENCHMARK_TEST_STEPS)
    parser.add_argument('--mem_sample_interval', type=int, default=BENCHMARK_MEM_SAMPLE_INTERVAL)
    args = parser.parse_args()
    tracker = BenchmarkTracker(gpu_id=0)
    mem_baseline = tracker.get_mem_used_mb()
    wp.init()
    scene_xml_path = Path(__file__).resolve().with_name('scene_unified_single_franka_fixed.xml')
    spec = mujoco.MjSpec.from_file(str(scene_xml_path))
    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    initial_positions = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
    for joint_name, pos in zip(joint_names, initial_positions):
        joint_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qposadr = mjm.jnt_qposadr[joint_id]
            mjd.qpos[qposadr] = pos
    for finger_joint in ['finger_joint1', 'finger_joint2']:
        joint_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, finger_joint)
        if joint_id >= 0:
            qposadr = mjm.jnt_qposadr[joint_id]
            mjd.qpos[qposadr] = 0.04
    mujoco.mj_forward(mjm, mjd)
    joint_limits = mjm.actuator_ctrlrange.copy()
    if mjm.nu != joint_limits.shape[0]:
        raise RuntimeError(f'Expected Franka nu=={joint_limits.shape[0]} actuators, got nu=={mjm.nu}. Scene: {scene_xml_path}')
    with wp.ScopedDevice('cuda:0'):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=args.num_envs, nconmax=args.nconmax, njmax=args.njmax)
        with wp.ScopedCapture() as capture:
            mjw.step(m, d)
        graph = capture.graph
        mem_post_init = tracker.get_mem_used_mb()
        for _ in range(args.warmup_steps):
            random_targets = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(args.num_envs, mjm.nu)).astype(np.float32)
            d.ctrl.assign(random_targets.reshape(-1))
            wp.capture_launch(graph)
        wp.synchronize()
        mem_post_warmup = tracker.get_mem_used_mb()
        peak_mem = mem_post_warmup
        start_time = time.perf_counter()
        for i in range(args.test_steps):
            random_targets = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(args.num_envs, mjm.nu)).astype(np.float32)
            d.ctrl.assign(random_targets.reshape(-1))
            wp.capture_launch(graph)
            wp.synchronize()
            if args.mem_sample_interval > 0 and i % args.mem_sample_interval == 0:
                peak_mem = max(peak_mem, tracker.get_mem_used_mb())
        total_time = time.perf_counter() - start_time
        mem_final = tracker.get_mem_used_mb()
        peak_mem = max(peak_mem, mem_final)
        fps = args.test_steps * args.num_envs / total_time
        result = {'num_envs': args.num_envs, 'fps': fps, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
        print(f'BENCHMARK_RESULT:{json.dumps(result)}')
if __name__ == '__main__':
    main()
