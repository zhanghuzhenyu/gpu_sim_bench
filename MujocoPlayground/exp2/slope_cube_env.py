import os
from typing import Dict, Any
import yaml
from ml_collections import ConfigDict
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

def get_config(config_path: str=None) -> ConfigDict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'slope_cube_config.yaml')
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ConfigDict(config_dict)
    if 'sim_dt' not in config:
        config.sim_dt = 0.002
    if 'ctrl_dt' not in config:
        config.ctrl_dt = 0.02
    if 'episode_length' not in config:
        config.episode_length = 1000
    return config

def load(config_path: str=None, config_overrides: Dict[str, Any]=None) -> 'SlopeCubeEnv':
    config = get_config(config_path)
    if config_overrides:
        config.update(config_overrides)
    return SlopeCubeEnv(config=config)

class SlopeCubeEnv(mjx_env.MjxEnv):

    def __init__(self, config: ConfigDict, config_overrides: Dict[str, Any]=None):
        if config_overrides:
            config.update(config_overrides)
        self._config = config
        xml_string = self._create_mjcf_scene()
        base_xml_path = os.path.join(os.path.dirname(__file__), 'base_scene.xml')
        with open(base_xml_path, 'r') as f:
            base_xml = f.read()
        full_xml = base_xml.replace('<!-- DYNAMIC_CONTENT -->', xml_string)
        self._mj_model = mujoco.MjModel.from_xml_string(full_xml)
        self._mjx_model = mjx.put_model(self._mj_model)
        self._sim_dt = config.sim_dt
        self._ctrl_dt = config.ctrl_dt
        self._mj_model.opt.timestep = self._sim_dt
        m = config.array.m
        self.num_cubes = m * m * m
        self._xml_path = None
        super().__init__(config, config_overrides)

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
    def config(self) -> ConfigDict:
        return self._config

    @property
    def action_size(self) -> int:
        return 0

    def _create_mjcf_scene(self) -> str:
        import numpy as np
        cfg = self._config
        ground = cfg.ground_plane
        slope = cfg.slope
        ball = cfg.ball
        cube = cfg.cube
        array = cfg.array
        angle_rad = np.deg2rad(slope.angle_deg)
        slope_length = float(slope.length)
        slope_width = float(slope.width)
        slope_thickness = float(slope.thickness)
        slope_center_x = float(slope.center_x)
        slope_center_y = float(slope.center_y)
        slope_center_z = float(slope.center_z)
        slope_bottom_z = slope_thickness / 2
        if hasattr(ball, 'center_x') and hasattr(ball, 'center_z'):
            ball_x = float(ball.center_x)
            ball_y = float(getattr(ball, 'center_y', slope_center_y))
            ball_z = float(ball.center_z)
        else:
            ball_x = slope_center_x
            ball_y = slope_center_y
            ball_z = slope_center_z + float(ball.radius)
        m = int(array.m)
        cube_size = float(cube.size)
        spacing = float(array.spacing)
        center_x = float(array.center_x)
        center_y = float(array.center_y)
        gp_length = float(ground.length)
        gp_width = float(ground.width)
        gp_thickness = float(ground.thickness)
        gp_cx = float(ground.center_x)
        gp_cy = float(ground.center_y)
        gp_cz = float(ground.center_z)
        xml_parts = []
        xml_parts.append(f'\n    <!-- Ground wood board (600mm x 600mm x 20mm) -->\n    <body name="ground_board" pos="{gp_cx} {gp_cy} {gp_cz}">\n      <geom name="ground_board_geom" type="box"\n            size="{gp_length / 2} {gp_width / 2} {gp_thickness / 2}"\n            rgba="0.7 0.5 0.3 1"\n            friction="{ground.static_friction} {ground.dynamic_friction} 0.0001"\n            solref="0.01 1"/>\n    </body>\n        ')
        xml_parts.append(f'\n    <!-- Slope -->\n    <body name="slope" pos="{slope_center_x} {slope_center_y} {slope_center_z}">\n      <geom name="slope_geom" type="box"\n            size="{slope_length / 2} {slope_width / 2} {slope_thickness / 2}"\n            euler="0 {slope.angle_deg} 0"\n            rgba="0.6 0.6 0.6 1"\n            friction="{slope.static_friction} {slope.dynamic_friction} 0.0001"\n            solref="0.01 1"/>\n    </body>\n        ')
        xml_parts.append(f'\n    <!-- Ball -->\n    <body name="ball" pos="{ball_x} {ball_y} {ball_z}">\n      <freejoint/>\n      <geom name="ball_geom" type="sphere"\n            size="{ball.radius}"\n            rgba="1.0 0.3 0.3 1"\n            mass="{ball.mass}"\n            friction="{ball.static_friction} {ball.dynamic_friction} 0.0001"\n            solref="0.01 1"/>\n      <inertial pos="0 0 0" mass="{ball.mass}"\n                diaginertia="{0.4 * ball.mass * ball.radius ** 2} {0.4 * ball.mass * ball.radius ** 2} {0.4 * ball.mass * ball.radius ** 2}"/>\n    </body>\n        ')
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    x = center_x + (i - (m - 1) / 2) * spacing
                    y = center_y + (j - (m - 1) / 2) * spacing
                    z = cube_size / 2 + k * spacing
                    cube_name = f'cube_{i}_{j}_{k}'
                    xml_parts.append(f'\n    <!-- Cube {i},{j},{k} -->\n    <body name="{cube_name}" pos="{x} {y} {z}">\n      <freejoint/>\n      <geom name="{cube_name}_geom" type="box"\n            size="{cube_size / 2} {cube_size / 2} {cube_size / 2}"\n            rgba="0.3 0.6 1.0 1"\n            mass="{cube.mass}"\n            friction="{cube.static_friction} {cube.dynamic_friction} 0.0001"\n            solref="0.01 1"/>\n      <inertial pos="0 0 0" mass="{cube.mass}"\n                diaginertia="{cube.mass * cube_size ** 2 / 6} {cube.mass * cube_size ** 2 / 6} {cube.mass * cube_size ** 2 / 6}"/>\n    </body>\n                    ')
        return '\n'.join(xml_parts)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        qpos = jp.array(self._mj_model.qpos0)
        qvel = jp.zeros(self.mjx_model.nv)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, impl=self.mjx_model.impl.value, nconmax=self._config.get('nconmax', 0), njmax=self._config.get('njmax', 0))
        data = mjx.forward(self.mjx_model, data)
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        metrics = {'ball_velocity': jp.linalg.norm(data.qvel[0:3]), 'ball_height': data.qpos[2], 'cubes_moved': 0.0}
        info = {}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        n_substeps = int(self._ctrl_dt / self._sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_substeps)
        obs = self._get_obs(data)
        ball_vel = jp.linalg.norm(data.qvel[0:3])
        reward = ball_vel * 0.01
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        ball_height = data.qpos[2]
        total_kinetic_energy = jp.sum(data.qvel ** 2)
        metrics = {'ball_velocity': jp.linalg.norm(data.qvel[0:3]), 'ball_height': ball_height, 'total_energy': total_kinetic_energy}
        return mjx_env.State(data, obs, reward, done, metrics, state.info)

    def _get_obs(self, data: mjx.Data) -> Dict[str, jax.Array]:
        ball_pos = data.qpos[0:7]
        ball_vel = data.qvel[0:6]
        cube_pos = data.qpos[7:]
        cube_vel = data.qvel[6:]
        obs = {'state': jp.concatenate([ball_pos, ball_vel, cube_pos, cube_vel])}
        return obs
