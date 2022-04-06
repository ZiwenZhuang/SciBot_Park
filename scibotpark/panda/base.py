import numpy as np
import os
from os import path as osp
from gym import spaces, Env

import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from scibotpark.pybullet.base import PybulletEnv
from scibotpark.panda.robot import PandaRobot

class PandaEnv(PybulletEnv, Env):
    def __init__(self,
            robot_kwargs= dict(),
            horizon= int(1e3),
            **kwargs,
        ):
        self.robot_kwargs = dict(
        ); self.robot_kwargs.update(robot_kwargs)
        self.horizon = horizon

        super().__init__(**kwargs)

    def _build_robot(self):
        self._robot = PandaRobot(pb_client= self.pb_client, **self.robot_kwargs)

    def _reset_robot(self, *args, **kwargs):
        self.robot.reset(*args, **kwargs)

    @property
    def action_space(self):
        cmd_limits = self.robot.get_cmd_limits()
        return spaces.Box(low= cmd_limits[0], high= cmd_limits[1])

    @property
    def observation_space(self):
        if self.robot.arm_control_mode == "joint":
            joint_limits = self.robot.get_joint_limits()
            return spaces.Box(joint_limits[0], joint_limits[1],)

        effector_state_space = np.array([np.inf] * 3 + [1] * 5)
        return spaces.Box(
            -effector_state_space,
            effector_state_space,
        ) # shape (7,) for translation, rotation, finger open/close

    def _get_obs(self):
        if self.robot.arm_control_mode == "joint":
            return self.robot.get_joint_states()

        poi_transform = self.robot.get_poi_world_transform()
        poi_euler = p.getEulerFromQuaternion(poi_transform[3:])
        finger_state = self.robot.get_finger_state()
        return np.concatenate([poi_transform[:3], poi_euler, [finger_state]])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.step_simulation_from_action(action)
        obs = self._get_obs()
        reward = 0.
        done = self.is_done(obs)
        info = dict()
        return obs, reward, done, info


if __name__ == "__main__":
    from scibotpark.panda.utils import get_keyboard_action
    import time
    # performing test
    env = PandaEnv(
        robot_kwargs= dict(
            arm_control_mode= "translation",
        ),
        nsubsteps= 1,
        pb_engine_parameter= dict(
            fixedTimeStep= 1./200.,
        ),
        pb_client= bullet_client.BulletClient(connection_mode=p.GUI),
    )
    env.robot.set_poi_visiblity(True)
    env.reset()

    action = np.zeros_like(env.action_space.sample())
    while True:
        action = get_keyboard_action(env, action)
        obs, reward, done, info = env.step(action)
        if done:
            print("Environment done, reset...")
            obs = env.reset()

        time.sleep(1./200. * 1)
