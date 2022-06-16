import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

import gym
from gym import spaces, Env
from scibotpark.pybullet.base import PybulletEnv
from scibotpark.utils.quick_args import save__init__args

class LocomotionEnv(PybulletEnv, Env):
    def __init__(self,
            RobotCls,
            robot_kwargs= dict(),
            render_kwargs= dict(),
            obs_types= ["joints"], # by default, contains "joints", "inertial",
            horizon= int(1e3),
            **kwargs,
        ):
        save__init__args(locals())
        super().__init__(**kwargs)

    def _build_robot(self):
        self._robot = self.RobotCls(pb_client= self.pb_client, **self.robot_kwargs)

    @property
    def action_space(self):
        limits = self.robot.get_cmd_limits()
        return spaces.Box(limits[0], limits[1], dtype= np.float32)
    
    @property
    def observation_space(self):
        obs_space = dict()
        if "joints" in self.obs_types:
            joint_limits_position = self.robot.get_joint_limits("position")
            joint_limits_velocity = self.robot.get_joint_limits("velocity")
            joint_limits = np.concatenate([joint_limits_position, joint_limits_velocity], axis= -1)
            obs_space["joints"] = spaces.Box(
                joint_limits[0],
                joint_limits[1],
                dtype= np.float32,
            )
        if "inertial" in self.obs_types:
            obs_space["inertial"] = spaces.Box(
                -np.inf,
                np.inf,
                shape= (12,), # linear / angular, velocity / position
                dtype= np.float32,
            )
        if len(self.obs_types) == 1:
            return obs_space[self.obs_types[0]]
        else:
            return spaces.Dict(obs_space)

    def _get_obs(self):
        obs = dict()
        if "joints" in self.obs_types:
            joint_positions = self.robot.get_joint_states("position")
            joint_velocities = self.robot.get_joint_states("velocity")
            obs["joints"] = np.concatenate([joint_positions, joint_velocities], axis= -1)
        if "inertial" in self.obs_types:
            inertial_data = self.robot.get_inertial_data()
            obs["inertial"] = np.concatenate([
                inertial_data["position"],
                inertial_data["linear_velocity"],
                inertial_data["rotation"],
                inertial_data["angular_velocity"],
            ])
        if len(self.obs_types) == 1:
            return obs[self.obs_types[0]]
        else:
            return obs

    def compute_reward(self, next_obs, **kwargs):
        return 0

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.step_simulation_from_action(action)

        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = self.is_done()
        info = {}
        return obs, reward, done, info
