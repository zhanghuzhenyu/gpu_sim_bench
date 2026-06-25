import argparse
import time
import torch
import numpy as np
import pynvml
import json
import gymnasium as gym
import sys
from pathlib import Path
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.utils.structs.pose import Pose
import sapien
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS, EXP1_CUBE_HALF_EXTENT, EXP1_CUBE_SPACING, EXP1_ENV_SPACING, EXP1_INITIAL_CENTER_Z, EXP1_NUM_CUBES_PER_DIM
try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_mem():
        return pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / 1024 / 1024
except Exception:
    import subprocess

    def get_mem():
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
            return float(result.strip())
        except Exception:
            return 0.0
mem_baseline = get_mem()

class CubeEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.cube_array_size = EXP1_NUM_CUBES_PER_DIM
        self.cube_spacing = EXP1_CUBE_SPACING
        super().__init__(*args, **kwargs)
        self._setup_minimal_action_space()

    def _setup_minimal_action_space(self):
        self._orig_single_action_space = gym.spaces.Box(low=np.array([], dtype=np.float32), high=np.array([], dtype=np.float32), shape=(0,))

    @property
    def _default_sim_config(self):
        return SimConfig(spacing=EXP1_ENV_SPACING, scene_config=SceneConfig())

    def _load_agent(self, options: dict):
        self.agent = None

    def _load_scene(self, options):
        m, spacing, center_z = (self.cube_array_size, self.cube_spacing, EXP1_INITIAL_CENTER_Z)
        half_n = (m - 1) / 2.0
        ground_builder = self.scene.create_actor_builder()
        ground_builder.add_box_collision(half_size=(50.0, 50.0, 0.05))
        ground_builder.initial_pose = Pose.create(sapien.Pose(p=[0.0, 0.0, -0.05]))
        ground_builder.build_static(name='ground')
        self.cubes = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    x, y, z = ((i - half_n) * spacing, (j - half_n) * spacing, center_z + (k - half_n) * spacing)
                    builder = self.scene.create_actor_builder()
                    builder.add_box_collision(half_size=(EXP1_CUBE_HALF_EXTENT, EXP1_CUBE_HALF_EXTENT, EXP1_CUBE_HALF_EXTENT), density=1.0)
                    builder.initial_pose = Pose.create(sapien.Pose(p=[x, y, z]))
                    self.cubes.append(builder.build(name=f'cube_{i}_{j}_{k}'))

    def _initialize_episode(self, env_idx, options):
        m, spacing, center_z = (self.cube_array_size, self.cube_spacing, EXP1_INITIAL_CENTER_Z)
        half_n = (m - 1) / 2.0
        idx = 0
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    x, y, z = ((i - half_n) * spacing, (j - half_n) * spacing, center_z + (k - half_n) * spacing)
                    state = torch.zeros((len(env_idx), 13), device=self.device)
                    state[:, 0] = x
                    state[:, 1] = y
                    state[:, 2] = z
                    state[:, 3] = 1.0
                    self.cubes[idx].set_state(state, env_idx=env_idx)
                    idx += 1

    def _get_obs_agent(self):
        return {}

    def evaluate(self):
        return {}

    def get_state_dict(self):
        return self.scene.get_sim_state()

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=1)
    args = parser.parse_args()
    env = CubeEnv(num_envs=args.num_envs, obs_mode='state', render_mode=None)
    env.reset()
    mem_post_init = get_mem()
    for _ in range(BENCHMARK_WARMUP_STEPS):
        env.step(None)
    torch.cuda.synchronize()
    mem_post_warmup = get_mem()
    test_steps, peak_mem = (BENCHMARK_TEST_STEPS, mem_post_warmup)
    start_time = time.perf_counter()
    for i in range(test_steps):
        env.step(None)
        if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
            peak_mem = max(peak_mem, get_mem())
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    mem_final = get_mem()
    peak_mem = max(peak_mem, mem_final)
    result = {'num_envs': args.num_envs, 'fps': test_steps * args.num_envs / total_time, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}')
    env.close()
if __name__ == '__main__':
    main()
