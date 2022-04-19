import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from gym import spaces, Env

from scibotpark.unitree.base import UnitreeEnv

class UnitreeLocomotionEnv(UnitreeEnv):
    def __init__(self,
            *args,
            include_vision_obs= False, # no obs_type
            forward_reward_ratio= 1,
            alive_reward_ratio= 1,
            torque_reward_ratio= 1,
            heading_reward_ratio= 1,
            alive_height_range= [0.2, 0.6],
            alive_row_pitch_limit= np.pi / 4, # in radians
            horizon= int(1e4),
            **kwargs,
        ):
        self.include_vision_obs = include_vision_obs
        self.forward_reward_ratio = forward_reward_ratio
        self.alive_reward_ratio = alive_reward_ratio
        self.torque_reward_ratio = torque_reward_ratio
        self.heading_reward_ratio = heading_reward_ratio
        self.alive_height_range = alive_height_range
        self.alive_row_pitch_limit = alive_row_pitch_limit
        self.horizon = horizon
        obs_type= ["joints", "inertial"]
        if self.include_vision_obs: obs_type.append("vision")

        super().__init__(*args, obs_type= obs_type, **kwargs)

        self.last_action = np.zeros(self.action_space.shape, dtype= np.float32)

    def reset(self, *args, **kwargs):
        return_ = super().reset(*args, **kwargs)
        self.last_action = np.zeros(self.action_space.shape, dtype= np.float32)
        return return_

    @property
    def observation_space(self):
        base_observation_space = super().observation_space
        observation_space = spaces.Box(
            low= np.concatenate([
                base_observation_space["joints"].low,
                base_observation_space["inertial"].low,
                self.action_space.low,
            ]),
            high= np.concatenate([
                base_observation_space["joints"].high,
                base_observation_space["inertial"].high,
                self.action_space.high,
            ]),
        )

        if self.include_vision_obs:
            observation_space = spaces.Dict(dict(
                vision= base_observation_space["vision"],
                proprioceptive= observation_space,
            ))

        return observation_space

    def _get_obs(self):
        base_obs = super()._get_obs()
        obs = np.concatenate([
            base_obs["joints"],
            base_obs["inertial"],
            self.last_action,
        ])

        if self.include_vision_obs:
            obs = dict(
                vision= base_obs["vision"],
                proprioceptive= obs,
            )

        return obs

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action

        self.step_simulation_from_action(action)

        obs = self._get_obs()

        reward, reward_info = self._compute_reward()
        done = self.is_done()
        info = reward_info
        return obs, reward, done, info

    def _compute_reward(self):
        # compute reward, which requies sophisticated computation at each timestep
        info = {}; reward = 0
        if self.alive_reward_ratio > 0:
            alive_reward = self.compute_alive_reward()
            reward += self.alive_reward_ratio * alive_reward
            info["alive_reward"] = alive_reward
        if self.forward_reward_ratio > 0:
            forward_reward = self.compute_forward_reward()
            reward += self.forward_reward_ratio * forward_reward
            info["forward_reward"] = forward_reward
        if self.torque_reward_ratio > 0:
            torque_reward = self.compute_torque_reward()
            reward += self.torque_reward_ratio * torque_reward
            info["torque_reward"] = torque_reward
        if self.heading_reward_ratio > 0:
            heading_reward = self.compute_heading_reward()
            reward += self.heading_reward_ratio * heading_reward
            info["heading_reward"] = heading_reward
        return reward, info

    def compute_alive_reward(self):
        inertial_data = self.robot.get_inertial_data()
        
        if inertial_data["position"][2] > self.alive_height_range[1] or inertial_data["position"][2] < self.alive_height_range[0]:
            return -1.
        
        rotation = inertial_data["rotation"]
        if np.abs(rotation[0]) > self.alive_row_pitch_limit or np.abs(rotation[1]) > self.alive_row_pitch_limit:
            return -1.
        
        return 1.

    def compute_forward_reward(self):
        inertial_data = self.robot.get_inertial_data()
        return inertial_data["linear_velocity"][0] # move towards +x direction

    def compute_torque_reward(self):
        torques = self.robot.get_joint_states("torque")
        return -np.power(np.linalg.norm(torques), 2) # negative of total torque norm

    def compute_heading_reward(self):
        inertial_data = self.robot.get_inertial_data()
        heading_reward = -np.abs(inertial_data["rotation"][2])
        return heading_reward

    def is_done(self):
        reached_horizon = super().is_done()
        dead = self.compute_alive_reward() < 0
        return reached_horizon or dead

if __name__ == "__main__":
    import time

    env = UnitreeLocomotionEnv(
        include_vision_obs= False, # no obs_type
        forward_reward_ratio= 1,
        alive_reward_ratio= 1,
        torque_reward_ratio= 1,
        alive_height_range= [0.2, 0.6],
        robot_kwargs= dict(
            robot_type= "a1",
            pb_control_mode= "DELTA_POSITION_CONTROL",
            pb_control_kwargs= dict(forces= [40] * 12),
            simulate_timestep= 1./500,
            default_base_transform= [0, 0, 0.42, 0, 0, 0, 1],
        ),
        connection_mode= p.GUI,
        bullet_debug= True,
    )
    obs = env.reset()

    while True:
        # env.debug_step()
        # env.pb_client.stepSimulation()
        # image = env.render(camera_name= "front")
        o, r, d, i = env.step(env.action_space.sample())

        if d:
            o = env.reset()

        time.sleep(1./500)

