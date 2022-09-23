import numpy as np
from PIL import Image, ImageDraw
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
            obs_types= ["joints", "inertial"], # by default, contains "joints", "inertial",
            horizon= int(1e3),
            simulate_timestep= 1./500,
            **kwargs,
        ):
        save__init__args(locals())
        super().__init__(**kwargs)
        self.default_camera_fov_kwargs = dict(
            fov= 60,
            nearVal= 0.01,
            farVal= 10,
        )

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
            ]).astype(np.float32)
        if len(self.obs_types) == 1:
            return obs[self.obs_types[0]]
        else:
            return obs

    def compute_reward(self, next_obs, **kwargs):
        return 0, dict()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.step_simulation_from_action(action)

        obs = self._get_obs()
        reward, reward_info = self.compute_reward(obs)
        done = self.is_done()
        info = {}; info.update(reward_info)
        self.current_info = info
        return obs, reward, done, info

    def render(self, mode= "vis", **render_kwargs):
        if "vis" in mode:
            """ Render array for recording visualization """
            robot_inertial = self.robot.get_inertial_data()
            view_matrix_kwargs = dict(
                distance= 0.8,
                roll= 0.,
                pitch= -30,
                yaw= 0.,
                upAxisIndex= 2, # z axis for up.
            ); view_matrix_kwargs.update(render_kwargs.get("view_matrix_kwargs", dict()))
            view_matrix_kwargs["yaw"] += robot_inertial["rotation"][2]
            image_data = self.pb_client.getCameraImage(
                width= render_kwargs.get("width", 320),
                height= render_kwargs.get("height", 240),
                viewMatrix= self.pb_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition= robot_inertial["position"],
                    **view_matrix_kwargs,
                ),
                projectionMatrix= self.pb_client.computeProjectionMatrixFOV(
                    aspect= render_kwargs.get("width", 320)/render_kwargs.get("height", 240),
                    **self.default_camera_fov_kwargs
                ),
            )
            np_image = image_data[2]
            if "dual_vis" in mode or "quad_vis" in mode:
                view_matrix_kwargs["yaw"] *= -1
                image_data_side = self.pb_client.getCameraImage(
                    width= render_kwargs.get("width", 320),
                    height= render_kwargs.get("height", 240),
                    viewMatrix= self.pb_client.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition= robot_inertial["position"],
                        **view_matrix_kwargs,
                    ),
                    projectionMatrix= self.pb_client.computeProjectionMatrixFOV(
                        aspect= render_kwargs.get("width", 320)/render_kwargs.get("height", 240),
                        **self.default_camera_fov_kwargs
                    ),
                )
                np_image = np.concatenate([image_data_side[2], np_image], axis= 1) # (H, 2W, C)
            if "quad_vis" in mode:
                image_datas = []
                view_matrix_kwargs["yaw"] += 180
                image_data_side = self.pb_client.getCameraImage(
                    width= render_kwargs.get("width", 320),
                    height= render_kwargs.get("height", 240),
                    viewMatrix= self.pb_client.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition= robot_inertial["position"],
                        **view_matrix_kwargs,
                    ),
                    projectionMatrix= self.pb_client.computeProjectionMatrixFOV(
                        aspect= render_kwargs.get("width", 320)/render_kwargs.get("height", 240),
                        **self.default_camera_fov_kwargs
                    ),
                )
                image_datas.append(image_data_side[2])
                view_matrix_kwargs["yaw"] *= -1
                image_data_side = self.pb_client.getCameraImage(
                    width= render_kwargs.get("width", 320),
                    height= render_kwargs.get("height", 240),
                    viewMatrix= self.pb_client.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition= robot_inertial["position"],
                        **view_matrix_kwargs,
                    ),
                    projectionMatrix= self.pb_client.computeProjectionMatrixFOV(
                        aspect= render_kwargs.get("width", 320)/render_kwargs.get("height", 240),
                        **self.default_camera_fov_kwargs
                    ),
                )
                image_datas.append(image_data_side[2])
                np_image = np.concatenate([
                    np_image,
                    np.concatenate(image_datas, axis= 1),
                ], axis= 0) # (2H, 2W, C)
            if "info" in mode and hasattr(self, "current_info"):
                # add reward info as text into np_image
                pil_image = Image.fromarray(np_image)
                pil_draw = ImageDraw.Draw(pil_image)
                for key_idx, key in enumerate(self.current_info.keys()):
                    pil_draw.text(
                        (16, 16 * (key_idx + 1)),
                        "{}: {:.3e}".format(key, self.current_info[key]),
                        (255, 0, 0),
                    )
                np_image = np.asarray(pil_image)
            return np_image
        else:
            raise NotImplementedError("Not implemented renderer for mode: {}".format(mode))
