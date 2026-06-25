import os
from typing import Any, Dict, Tuple
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_lib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_defaults import EXP1_CUBE_HALF_EXTENT, EXP1_CUBE_MASS, EXP1_CUBE_SPACING, EXP1_DT, EXP1_INITIAL_CENTER_Z, EXP1_NUM_CUBES_PER_DIM

def get_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(dict(ctrl_dt=EXP1_DT, sim_dt=EXP1_DT, episode_length=1000, action_repeat=1, num_cubes_per_dim=EXP1_NUM_CUBES_PER_DIM, cube_size=EXP1_CUBE_HALF_EXTENT, cube_spacing=EXP1_CUBE_SPACING, initial_height=EXP1_INITIAL_CENTER_Z, vision=False, impl='jax'))

def _create_mjcf_with_cubes(num_cubes_per_dim: int, cube_size: float, cube_spacing: float, initial_height: float) -> str:
    current_dir = os.path.dirname(__file__)
    base_xml_path = os.path.join(current_dir, 'cube_array.xml')
    with open(base_xml_path, 'r') as f:
        base_xml = f.read()
    cube_bodies = []
    cube_id = 0
    for i in range(num_cubes_per_dim):
        for j in range(num_cubes_per_dim):
            for k in range(num_cubes_per_dim):
                half_n = (num_cubes_per_dim - 1) / 2.0
                x = (i - half_n) * cube_spacing
                y = (j - half_n) * cube_spacing
                z = initial_height + (k - half_n) * cube_spacing
                r = 0.3 + i / num_cubes_per_dim * 0.6
                g = 0.3 + j / num_cubes_per_dim * 0.6
                b = 0.3 + k / num_cubes_per_dim * 0.6
                cube_body = f'\n    <body name="cube_{cube_id}" pos="{x} {y} {z}">\n      <freejoint/>\n      <geom name="cube_geom_{cube_id}" type="box" size="{cube_size} {cube_size} {cube_size}"\n            rgba="{r} {g} {b} 1" mass="{EXP1_CUBE_MASS}" condim="3" friction="1 0.005 0.0001"/>\n    </body>'
                cube_bodies.append(cube_body)
                cube_id += 1
    cubes_xml = '\n'.join(cube_bodies)
    modified_xml = base_xml.replace('</worldbody>', f'{cubes_xml}\n  </worldbody>')
    return modified_xml

class CubeArrayEnv(mjx_env.MjxEnv):

    def __init__(self, config: config_dict.ConfigDict=None, config_overrides=None):
        if config is None:
            config = get_config()
        super().__init__(config, config_overrides)
        mjcf_string = _create_mjcf_with_cubes(self._config.num_cubes_per_dim, self._config.cube_size, self._config.cube_spacing, self._config.initial_height)
        self._mj_model = mujoco.MjModel.from_xml_string(mjcf_string)
        self._mj_model.opt.timestep = self._config.sim_dt
        if self._config.impl == 'jax':
            self._mjx_model = mjx.put_model(self._mj_model)
        else:
            raise ValueError(f'Unknown impl: {self._config.impl}')
        self._action_size = 0
        self._xml_path = None
        self.num_cubes = self._config.num_cubes_per_dim ** 3

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def config(self) -> config_dict.ConfigDict:
        return self._config

    @property
    def action_size(self) -> int:
        return self._action_size

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        qpos = jp.array(self._mj_model.qpos0)
        qvel = jp.zeros(self.mjx_model.nv)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, impl=self.mjx_model.impl.value, nconmax=self._config.get('nconmax', 0), njmax=self._config.get('njmax', 0))
        data = mjx.forward(self.mjx_model, data)
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        metrics = {'total_energy': self._compute_total_energy(data)}
        info = {}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        n_substeps = int(self._ctrl_dt / self._sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_substeps)
        obs = self._get_obs(data)
        cube_heights = self._get_cube_heights(data)
        reward = -jp.var(cube_heights)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        metrics = {}
        return mjx_env.State(data, obs, reward, done, metrics, state.info)

    def _get_obs(self, data: mjx.Data) -> Dict[str, jax.Array]:
        n_cubes = self.num_cubes
        if data.qpos.ndim == 1:
            qpos_reshaped = data.qpos.reshape(n_cubes, 7)
            positions = qpos_reshaped[:, :3]
            qvel_reshaped = data.qvel.reshape(n_cubes, 6)
            velocities = qvel_reshaped[:, 3:6]
            obs = {'state': jp.concatenate([positions.reshape(n_cubes * 3), velocities.reshape(n_cubes * 3)])}
        else:
            qpos_reshaped = data.qpos.reshape(-1, n_cubes, 7)
            positions = qpos_reshaped[:, :, :3]
            qvel_reshaped = data.qvel.reshape(-1, n_cubes, 6)
            velocities = qvel_reshaped[:, :, 3:6]
            obs = {'state': jp.concatenate([positions.reshape(-1, n_cubes * 3), velocities.reshape(-1, n_cubes * 3)], axis=-1)}
        return obs

    def _get_cube_heights(self, data: mjx.Data) -> jax.Array:
        n_cubes = self.num_cubes
        if data.qpos.ndim == 1:
            qpos_reshaped = data.qpos.reshape(n_cubes, 7)
            heights = qpos_reshaped[:, 2]
        else:
            qpos_reshaped = data.qpos.reshape(-1, n_cubes, 7)
            heights = qpos_reshaped[:, :, 2]
        return heights

    def _compute_total_energy(self, data: mjx.Data) -> jax.Array:
        ke = 0.5 * jp.sum(data.qvel ** 2)
        heights = self._get_cube_heights(data)
        pe = jp.sum(heights) * 9.81 * EXP1_CUBE_MASS
        return ke + pe

def load(config: config_dict.ConfigDict=None, config_overrides: Dict[str, Any]=None) -> CubeArrayEnv:
    if config is None:
        config = get_config()
    return CubeArrayEnv(config, config_overrides)
