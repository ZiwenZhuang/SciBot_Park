import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client
from gym import spaces

from scibotpark.locomotion.base import LocomotionEnv
from scibotpark.quadruped_robot.meta_quadruped.robot import MetaQuadrupedRobot
from scibotpark.quadruped_robot.meta_quadruped.config_sampler import MetaQuadruped_ConfigSampler
from scibotpark.utils.quick_args import save__init__args

class MetaQuadrupedForward(LocomotionEnv):
    """ This is a meta env, which randomly samples robot configuration and surroundings for training.
    """
    def __init__(self,
            RobotCls= MetaQuadrupedRobot, # must be a subclass of MetaQuadrupedRobot
            robot_kwargs= dict(),
            RobotConfigSampler= MetaQuadruped_ConfigSampler,
            robot_config_sampler_kwargs= dict(),
            reward_ratios= dict(),
            resample_on_reset= False,
            alive_row_pitch_limit= np.pi/4,
            **kwargs,
        ):
        save__init__args(locals())
        self.robot_config_sampler = RobotConfigSampler(**robot_config_sampler_kwargs)
        self.robot_kwargs = self.robot_config_sampler.sample(self.robot_kwargs)
        super().__init__(RobotCls= RobotCls, robot_kwargs= self.robot_kwargs, **kwargs)
        self.set_max_height()
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
        return observation_space

    def _get_obs(self):
        base_obs = super()._get_obs()
        obs = np.concatenate([
            base_obs["joints"],
            base_obs["inertial"],
            self.last_action,
        ])
        return obs

    def set_max_height(self):
        """ Compute the maximum base height based on the robot, in order to prevent the robot from
        jumping out of the ground.
        """
        thigh_length = self.robot.configuration["thigh_size"][1]
        shin_length = self.robot.configuration["shin_size"][1]
        self.max_height = thigh_length + shin_length

    def _reset_robot(self, *args, **kwargs):
        if self.resample_on_reset:
            self.robot_kwargs = self.robot_config_sampler.sample(self.robot_kwargs)
            delattr(self, "_robot")
            self._build_robot()
            self.set_max_height()
        return super()._reset_robot(*args, **kwargs)

    def compute_reward(self, *args, **kwargs):
        # compute reward, which requies sophisticated computation at each timestep
        info = {}; reward = 0
        if self.reward_ratios.get("alive_reward", 0) > 0:
            alive_reward = self.compute_alive_reward()
            reward += self.reward_ratios.get("alive_reward", 0) * alive_reward
            info["alive_reward"] = alive_reward
        if self.reward_ratios.get("forward_reward", 0) > 0:
            forward_reward = self.compute_forward_reward()
            reward += self.reward_ratios.get("forward_reward", 0) * forward_reward
            info["forward_reward"] = forward_reward
        if self.reward_ratios.get("torque_reward", 0) > 0:
            torque_reward = self.compute_torque_reward()
            reward += self.reward_ratios.get("torque_reward", 0) * torque_reward
            info["torque_reward"] = torque_reward
        if self.reward_ratios.get("heading_reward", 0) > 0:
            heading_reward = self.compute_heading_reward()
            reward += self.reward_ratios.get("heading_reward", 0) * heading_reward
            info["heading_reward"] = heading_reward
        return reward, info

    def compute_alive_reward(self):
        inertial_data = self.robot.get_inertial_data()
        
        if inertial_data["position"][2] > self.max_height:
            return -1.
        
        if len(self.pb_client.getContactPoints(self.robot.body_id, self.plane_id, -1, -1)) > 0:
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

    def step(self, action):
        self.last_action = action
        return super().step(action)
        

if __name__ == "__main__":
    import time
    env = MetaQuadrupedForward(
        robot_kwargs= dict(),
        robot_config_sampler_kwargs= dict(
        ),
        reward_ratios= dict(
            alive_reward= 1.,
            forward_reward= 1.,
            torque_reward= 1.,
            heading_reward= 1.,
        ),
        resample_on_reset= True,
        pb_client= bullet_client.BulletClient(connection_mode= p.GUI),
    )
    obs = env.reset()
    action_space = env.action_space
    while True:
        action = action_space.sample()
        obs, reward, done, info = env.step(action)
        time.sleep(1/250)
        if done:
            obs = env.reset()
