import argparse
import time
import torch
import numpy as np
import pynvml
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import BENCHMARK_MEM_SAMPLE_INTERVAL, BENCHMARK_TEST_STEPS, BENCHMARK_WARMUP_STEPS, EXP1_CUBE_EDGE, EXP1_CUBE_MASS, EXP1_CUBE_SPACING, EXP1_DT, EXP1_ENV_SPACING, EXP1_INITIAL_CENTER_Z, EXP1_NUM_CUBES_PER_DIM
try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_mem():
        return pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / 1024 / 1024
except Exception:

    def get_mem():
        return 0.0
mem_baseline = get_mem()
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

def dummy_obs(env):
    return torch.zeros(env.num_envs, 1, device=env.device)

@configclass
class CubeSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path='/World/ground', terrain_type='plane')

    def __post_init__(self):
        m = EXP1_NUM_CUBES_PER_DIM
        spacing = EXP1_CUBE_SPACING
        center_z = EXP1_INITIAL_CENTER_Z
        half_n = (m - 1) / 2.0
        for x in range(m):
            for y in range(m):
                for z in range(m):
                    px = (x - half_n) * spacing
                    py = (y - half_n) * spacing
                    pz = center_z + (z - half_n) * spacing
                    setattr(self, f'cube_{x}_{y}_{z}', RigidObjectCfg(prim_path=f'{{ENV_REGEX_NS}}/cube_{x}_{y}_{z}', spawn=sim_utils.CuboidCfg(size=(EXP1_CUBE_EDGE, EXP1_CUBE_EDGE, EXP1_CUBE_EDGE), rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=EXP1_CUBE_MASS), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, pz))))

@configclass
class ObsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        time = ObservationTermCfg(func=dummy_obs)
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionCfg:
    pass

@configclass
class CubeEnvCfg(ManagerBasedEnvCfg):
    scene = CubeSceneCfg(num_envs=args_cli.num_envs, env_spacing=EXP1_ENV_SPACING)
    observations = ObsCfg()
    actions = ActionCfg()

    def __post_init__(self):
        self.sim.dt = EXP1_DT
        self.decimation = 1

def main():
    env = ManagerBasedEnv(cfg=CubeEnvCfg())
    env.reset()
    mem_post_init = get_mem()
    for _ in range(BENCHMARK_WARMUP_STEPS):
        env.step(torch.zeros(env.num_envs, 0, device=env.device))
    torch.cuda.synchronize()
    mem_post_warmup = get_mem()
    test_steps = BENCHMARK_TEST_STEPS
    peak_mem = mem_post_warmup
    start_time = time.perf_counter()
    for i in range(test_steps):
        env.step(torch.zeros(env.num_envs, 0, device=env.device))
        if i % BENCHMARK_MEM_SAMPLE_INTERVAL == 0:
            peak_mem = max(peak_mem, get_mem())
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    mem_final = get_mem()
    peak_mem = max(peak_mem, mem_final)
    fps = test_steps * args_cli.num_envs / total_time
    result = {'num_envs': args_cli.num_envs, 'fps': fps, 'mem_baseline': mem_baseline, 'mem_post_init': mem_post_init, 'mem_post_warmup': mem_post_warmup, 'mem_peak': peak_mem, 'mem_final': mem_final}
    print(f'BENCHMARK_RESULT:{json.dumps(result)}')
    simulation_app.close()
if __name__ == '__main__':
    main()
