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
            alive_height_ratio= (0.4, 1.2),
            dead_penalty= 20., # alive reward without factor is 1. But the dead penalty should be tunable
            **kwargs,
        ):
        save__init__args(locals())
        self.robot_config_sampler = RobotConfigSampler(**robot_config_sampler_kwargs)
        self.robot_kwargs = self.robot_config_sampler.sample(self.robot_kwargs)
        super().__init__(RobotCls= RobotCls, robot_kwargs= self.robot_kwargs, **kwargs)
        self.set_height_limit()
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
            dtype= np.float32
        )
        return observation_space

    def _get_obs(self):
        base_obs = super()._get_obs()
        obs = np.concatenate([
            base_obs["joints"],
            base_obs["inertial"],
            self.last_action,
        ]).astype(np.float32)
        return obs

    def set_height_limit(self):
        """ Compute the maximum base height based on the robot, in order to prevent the robot from
        jumping out of the ground.
        """
        thigh_length = self.robot.configuration["thigh_size"][1]
        shin_length = self.robot.configuration["shin_size"][1]
        self.alive_height_range = np.array((thigh_length + shin_length,), dtype= np.float).repeat(2,)
        self.alive_height_range[0] *= self.alive_height_ratio[0]
        self.alive_height_range[1] *= self.alive_height_ratio[1]

    def _build_robot(self):
        super()._build_robot()
        self.robot_kwargs["default_base_transform"] = np.zeros((7,), dtype= np.float32)
        
        hip_radius = self.robot.configuration["hip_size"][0]
        thigh_radius = self.robot.configuration["thigh_size"][0]
        thigh_length = self.robot.configuration["thigh_size"][1]
        shin_length = self.robot.configuration["shin_size"][1]
        # foot_size = self.robot.configuration["foot_size"]
        default_height = 2 * np.sqrt(1/3) * np.sqrt(1/2) * (thigh_length + shin_length) - hip_radius - thigh_radius * 2
        self.robot_kwargs["default_base_transform"][2] = default_height
        self.robot_kwargs["default_base_transform"][6] = 1

    def _reset_robot(self, *args, **kwargs):
        if self.resample_on_reset:
            self.robot_kwargs = self.robot_config_sampler.sample(self.robot_kwargs)
            delattr(self, "_robot")
            self._build_robot()
            self.set_height_limit()
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
        if self.reward_ratios.get("sliding_reward", 0) > 0:
            sliding_reward = self.compute_sliding_reward()
            reward += self.reward_ratios.get("sliding_reward", 0) * sliding_reward
            info["sliding_reward"] = sliding_reward
        if self.reward_ratios.get("torque_reward", 0) > 0:
            torque_reward = self.compute_torque_reward()
            reward += self.reward_ratios.get("torque_reward", 0) * torque_reward
            info["torque_reward"] = torque_reward
        if self.reward_ratios.get("heading_reward", 0) > 0:
            heading_reward = self.compute_heading_reward()
            reward += self.reward_ratios.get("heading_reward", 0) * heading_reward
            info["heading_reward"] = heading_reward
        if self.reward_ratios.get("spinning_reward", 0) > 0:
            spinning_reward = self.compute_spinning_reward()
            reward += self.reward_ratios.get("spinning_reward", 0) * spinning_reward
            info["spinning_reward"] = spinning_reward
        if self.reward_ratios.get("joint_velocity_reward", 0) > 0:
            joint_velocity_reward = self.compute_joint_velocity_reward()
            reward += self.reward_ratios.get("joint_velocity_reward", 0) * joint_velocity_reward
            info["joint_velocity_reward"] = joint_velocity_reward
        if self.reward_ratios.get("action_reward", 0) > 0:
            action_reward = self.compute_action_reward()
            reward += self.reward_ratios.get("action_reward", 0) * action_reward
            info["action_reward"] = action_reward
        return reward, info

    def compute_alive_reward(self):
        inertial_data = self.robot.get_inertial_data()
        hip_ids = [0, 6, 12, 18]
        
        # check if any hip get higher than max_height
        for hip_id in hip_ids:
            hip_position = self.pb_client.getLinkState(self.robot.body_id, hip_id)[4]
            if hip_position[2] > self.alive_height_range[1] or hip_position[2] < self.alive_height_range[0]:
                return -self.dead_penalty
        
        # check if base touches the ground
        if len(self.pb_client.getContactPoints(self.robot.body_id, self.plane_id, -1, -1)) > 0:
            return -self.dead_penalty
        # check if hips touch the ground
        for hip_id in hip_ids:
            if len(self.pb_client.getContactPoints(self.robot.body_id, self.plane_id, hip_id, -1)) > 0:
                return -self.dead_penalty
        
        rotation = inertial_data["rotation"]
        if np.abs(rotation[0]) > self.alive_row_pitch_limit or np.abs(rotation[1]) > self.alive_row_pitch_limit:
            return -self.dead_penalty
        
        return 1.

    def compute_forward_reward(self):
        inertial_data = self.robot.get_inertial_data()
        return inertial_data["linear_velocity"][0] # move towards +x direction

    def compute_sliding_reward(self):
        """ This is actually a penalty """
        inertial_data = self.robot.get_inertial_data()
        return - (inertial_data["linear_velocity"][1])**2 - (inertial_data["linear_velocity"][2])**2

    def compute_torque_reward(self):
        torques = self.robot.get_joint_states("torque")
        return -np.power(np.linalg.norm(torques), 2) # negative of total torque norm

    def compute_heading_reward(self):
        inertial_data = self.robot.get_inertial_data()
        heading_reward = -np.abs(inertial_data["rotation"][2])
        return heading_reward

    def compute_spinning_reward(self):
        """ This is actually a penalty """
        inertial_data = self.robot.get_inertial_data()
        return - (inertial_data["angular_velocity"][0])**2 - (inertial_data["angular_velocity"][1])**2
    
    def compute_joint_velocity_reward(self):
        """ This is actually a penalty
        There is no joint acceleration in Pybullet, but ETH's Learn to walk in minutes paper used acceleration
        """
        velocity = self.robot.get_joint_states("velocity")
        return -np.power(np.linalg.norm(velocity), 2) # negative of total joint velocity norm

    def compute_action_reward(self):
        """ This is a penalty term. NOTE: Please check whether the environment action is all
        relative command (e.g. position delta, velocity, torque). It has to be this.
        """
        return -np.power(np.linalg.norm(self.last_action), 2)

    def is_done(self):
        reached_horizon = super().is_done()
        dead = self.compute_alive_reward() < 0
        return reached_horizon or dead

    def step(self, action):
        self.last_action = action
        o, r, d, i = super().step(action)
        i.update(dict(timeout= super().is_done()))
        return o, r, d, i
        
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    clients = [bullet_client.BulletClient(connection_mode= p.DIRECT) for _ in range(1)]
    env = MetaQuadrupedForward(
        robot_kwargs= dict(),
        robot_config_sampler_kwargs= dict(
            base_mass_range= (10., 25.0), # kg
            base_length_range= (0.16, 0.25), # meter, x axis
            base_width_range= (0.02, 0.08), # meter, y axis
            base_height_range= (0.02, 0.08), # meter, z axis
            hip_radius_range= (0.006, 0.01), # meter,
            thigh_mass_range= (0.1, 0.2), # kg
            thigh_length_range= (0.12, 0.16), # meter,
            shin_mass_range= (0.1, 0.2), # kg
            shin_length_range= (0.12, 0.16), # meter,
            foot_mass_range= (0.01, 0.1), # kg
            foot_size_range= (0.01, 0.02), # meter,
            motor_torque_range= (25, 40), # Nm
        ),
        reward_ratios= dict(
            alive_reward= 1.,
            forward_reward= 1.,
            torque_reward= 1.,
            heading_reward= 1.,
        ),
        resample_on_reset= True,
        nsubsteps= 10,
        pb_client= bullet_client.BulletClient(connection_mode= p.GUI),
    )
    obs = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    observation_space.contains(obs)
    print("action: {}".format(action_space))
    print("obs: {}".format(observation_space))
    while True:
        action = np.zeros(action_space.shape)
        # action = action_space.sample()
        obs, reward, done, info = env.step(action)
        img = env.render(
            mode= "vis",
            width= 320,
            height= 240,
            view_matrix_kwargs= dict(
                distance= 0.8,
                roll= 0.,
                pitch= -30,
                yaw= -60.,
                upAxisIndex= 2, # z axis for up.
            ),
        )
        # plt.imshow(img); plt.show()
        time.sleep(1/5)
        if done:
            obs = env.reset()
            time.sleep(3/10)
        print(obs[:3], end= "\r")
