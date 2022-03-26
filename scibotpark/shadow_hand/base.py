import numpy as np
from gym import spaces, Env

import pybullet

from scibotpark.pybullet.base import PybulletEnv
from scibotpark.shadow_hand.robot import ShadowHandRobot

class ShadowHandEnv(PybulletEnv, Env):
    def __init__(self,
            robot_kwargs= dict(),
            **kwargs,
        ):
        self.robot_kwargs = robot_kwargs

        super().__init__(**kwargs)

    def _build_robot(self):
        self._robot = ShadowHandRobot(
            pb_client= self.pb_client,
            **self.robot_kwargs
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
        limits = self.robot.get_joint_limits()
        return spaces.Box(
            low= limits[0],
            high= limits[1],
            dtype= np.float32,
        )

    def _get_obs(self):
        return self.robot.get_joint_states()

    def _compute_reward(self, obs):
        return 0.

    def get_hand_reached_positions(self):
        """ return current all finger reached positions """
        hand_links_positions = [
            lp[0] for lp in self.pb_client.getLinkStates(self.robot.body_id, list(range(self.hand_num_joints)))
        ] # (n_joints, 3)
        return np.array(hand_links_positions)
    
    def render(self, mode= "rgb_array", width= 640, height= 480,
            cameraTargetPosition= None,
            distance= 0.4,
            roll= 0.,
            pitch= -30,
            yaw= -135,
            **kwargs
        ):
        """ Designed only for visualization """
        image = self.pb_client.getCameraImage(
            width= width,
            height= height,
            viewMatrix= pybullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition= [0, 0, self.robot.init_forearm_position[2]/2] if cameraTargetPosition is None else cameraTargetPosition,
                distance= distance,
                roll= roll,
                pitch= pitch,
                yaw= yaw,
                upAxisIndex= 2,
            ),
            projectionMatrix= pybullet.computeProjectionMatrixFOV(
                fov= 60,
                aspect= float(width) / height,
                nearVal= 0.1,
                farVal= 100.0,
            ),
        )
        return image[2].reshape(height, width, 4)[:, :, :3] # (width, height, 3)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.step_simulation_from_action(action)

        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = False
        info = {}
        return obs, reward, done, info