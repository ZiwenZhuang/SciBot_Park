import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

import gym
from gym import spaces, Env

from scibotpark.unitree.robot import UniTreeRobot
from scibotpark.pybullet.base import PybulletEnv

class UnitreeEnv(PybulletEnv, Env):
    def __init__(self,
            obs_type= ["vision"], # a list contains: "vision", "joints", "inertial"
            render_kwargs= dict(), # for getting visual observation
            robot_kwargs= dict(),
            connection_mode= p.DIRECT,
            bullet_debug= False,
            **kwargs,
        ):
        self.obs_type = obs_type
        self.render_kwargs = dict(
            resolution= (480, 480),
            camera_name= "front",
            modal= "rgb",
        ); self.render_kwargs.update(render_kwargs)
        self.robot_kwargs = dict(
            robot_type= "a1",
            default_base_transform= np.array([0, 0, 0.5, 0, 0, 0, 1]),
            pb_control_mode= "DELTA_POSITION_CONTROL", # "POSITION_CONTROL", "VELOCITY_CONTROL", "TORQUE_CONTROL", "DELTA_POSITION_CONTROL"
            pb_control_kwargs= dict(forces= [40] * 12),
        ); self.robot_kwargs.update(robot_kwargs)
        if self.robot_kwargs["pb_control_mode"] == "TORQUE_CONTROL" or self.robot_kwargs["pb_control_mode"] == p.TORQUE_CONTROL:
            self.robot_kwargs["pb_control_kwargs"].pop("forces")
        self.bullet_debug = bullet_debug

        pb_client = bullet_client.BulletClient(connection_mode= connection_mode, options= "")
        super().__init__(pb_client= pb_client, **kwargs)
        
    def _build_robot(self):
        self._robot = UniTreeRobot(
            bullet_debug= self.bullet_debug,
            pb_client= self.pb_client,
            **self.robot_kwargs,
        )

    @property
    def action_space(self):
        limits = self.robot.get_cmd_limits()
        return spaces.Box(
            limits[0],
            limits[1],
            dtype= np.float32,
        )

    @property
    def observation_space(self):
        obs_space = dict()
        if "vision" in self.obs_type:
            obs_space["vision"] = spaces.Box(
                0, 1,
                shape= (*self.visual_resolution, 3),
                dtype= np.float32,
            )
        if "joints" in self.obs_type:
            joint_limits_position = self.robot.get_joint_limits("position")
            joint_limits_velocity = self.robot.get_joint_limits("velocity")
            joint_limits = np.concatenate([joint_limits_position, joint_limits_velocity], axis= -1)
            obs_space["joints"] = spaces.Box(
                joint_limits[0],
                joint_limits[1],
                dtype= np.float32,
            )
        if "inertial" in self.obs_type:
            obs_space["inertial"] = spaces.Box(
                -np.inf,
                np.inf,
                shape= (12,), # linear / angular, velocity / position
                dtype= np.float32,
            )
        if len(self.obs_type) == 1:
            return obs_space[self.obs_type[0]]
        else:
            return spaces.Dict(obs_space)

    def _get_obs(self):
        obs = dict()
        if "vision" in self.obs_type:
            obs["vision"] = self.render(**self.render_kwargs)
        if "joints" in self.obs_type:
            joint_positions = self.robot.get_joint_states("position")
            joint_velocities = self.robot.get_joint_states("velocity")
            obs["joints"] = np.concatenate([joint_positions, joint_velocities], axis= -1)
        if "inertial" in self.obs_type:
            inertial_data = self.robot.get_inertial_data()
            obs["inertial"] = np.concatenate([
                inertial_data["position"],
                inertial_data["linear_velocity"],
                inertial_data["rotation"],
                inertial_data["angular_velocity"],
            ])
        if len(self.obs_type) == 1:
            return obs[self.obs_type[0]]
        else:
            return obs
        
    def _reset_robot(self):
        self.robot.reset(self.robot_kwargs["default_base_transform"])

    def debug_step(self):
        self.robot.send_cmd_from_bullet_debug()
        self.pb_client.stepSimulation()

    def render(self, **render_kwargs):
        if "mode" in render_kwargs and render_kwargs["mode"] == "rgb_array":
            kwargs = self.render_kwargs.copy()
            kwargs["resolution"] = (render_kwargs["width"], render_kwargs["height"])
            return self.robot.get_onboard_camera_image(**kwargs)
        else:
            return self.robot.get_onboard_camera_image(**render_kwargs)

    def compute_reward(self, next_obs, **kwargs):
        return 0

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.step_simulation_from_action(action)

        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = False
        info = {}
        return obs, reward, done, info

if __name__ == "__main__":
    import tqdm, time

    env = UnitreeEnv(
        obs_type= "vision",
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(forces= [40] * 12),
            simulate_timestep= 1./500,
        ),
        connection_mode= p.GUI,
        bullet_debug= True,
    )
    obs = env.reset()

    for _ in tqdm.tqdm(range(10000)):
        # env.debug_step()
        # env.pb_client.stepSimulation()
        # image = env.render(camera_name= "front")
        obs = env.step(env.action_space.sample())[0]

        time.sleep(1./500)
