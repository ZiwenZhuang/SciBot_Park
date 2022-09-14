import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from PIL import Image, ImageDraw
from gym import spaces

from scibotpark.locomotion.meta_robot import MetaQuadrupedForward
from scibotpark.locomotion.noisy_sensor import NoisySensorMixin
from scibotpark.locomotion.perturbation import RobotBasePerturbationMixin

class RobustMetaQuadrupedLocomotion(NoisySensorMixin, RobotBasePerturbationMixin, MetaQuadrupedForward):
    """ Different from MetaQuadrupedForward, this class added the moving command """
    def __init__(self, *args,
            moving_max_speeds= (0.1, 0.1, 0.2), # x, y, raw expectation speed
            binary_move_cmd= False, # if true, the cmd will be only max speed or zero.
            move_prob= 0.9, # the other option is to let the robot stand still
            forward_reward_max= 1., # the actual forward reward will be max - norm(speed diff)
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.moving_max_speeds = moving_max_speeds if isinstance(moving_max_speeds, np.ndarray) else np.array(moving_max_speeds)
        self.binary_move_cmd = binary_move_cmd
        self.move_prob = move_prob
        self.forward_reward_max = forward_reward_max
        self.moving_cmd = np.zeros((3,), dtype= np.float32) # resample on each reset

    @property
    def observation_space(self):
        base_observation_space = super().observation_space
        if isinstance(base_observation_space, dict):
            return dict(
                command= spaces.Box(
                    -self.moving_max_speeds,
                    self.moving_max_speeds,
                    dtype= np.float32,
                ),
                **base_observation_space,
            )
        else:
            return spaces.Box(
                low= np.concatenate([
                    -self.moving_max_speeds,
                    base_observation_space.low,
                ]),
                high= np.concatenate([
                    self.moving_max_speeds,
                    base_observation_space.high,
                ]),
                dtype= np.float32,
            )

    def _get_obs(self):
        base_obs = super()._get_obs()
        if isinstance(base_obs, dict):
            return dict(
                command= self.moving_cmd,
                **base_obs,
            )
        else:
            return np.concatenate([
                self.moving_cmd,
                base_obs,
            ])

    def step_simulation_from_action(self, action):
        physics_engine_parameters = self.pb_client.getPhysicsEngineParameters()
        n_seconds_to_pass = physics_engine_parameters["fixedTimeStep"] * self.nsubsteps
        self.expected_heading += n_seconds_to_pass * self.moving_cmd[2]
        return super().step_simulation_from_action(action)

    def step(self, action):
        o, r, d, i = super().step(action)
        i["expected_heading"] = self.expected_heading
        i["cmd_x"] = self.moving_cmd[0]
        i["cmd_y"] = self.moving_cmd[1]
        return o, r, d, i

    def render(self, mode="vis", **render_kwargs):
        return_ = super().render(mode, **render_kwargs)
        if "vis" in mode:
            pil_image = Image.fromarray(return_)
            pil_draw = ImageDraw.Draw(pil_image)
            pil_draw.text(
                (16, 16),
                "cmd: {:.2f}, {:.2f}, {:.2f}".format(*self.moving_cmd),
                (255, 0, 0),
            )
            return_ = np.asarray(pil_image)
        return return_

    def reset(self, *args, **kwargs):
        if self.binary_move_cmd:
            if np.random.uniform() < self.move_prob:
                self.moving_cmd = np.random.randint(low= 0, high= 2, size= (3,),).astype(np.float32) * 2 - 1
                self.moving_cmd *= self.moving_max_speeds
            else:
                self.moving_cmd = np.zeros((3,))
        else:
            self.moving_cmd = np.random.uniform(
                low= -self.moving_max_speeds,
                high= self.moving_max_speeds,
            ) if np.random.uniform() < self.move_prob else np.array((0, 0, 0), dtype= np.float32)
        self.expected_heading = 0.
        return super().reset(*args, **kwargs)

    @staticmethod
    def phi(x):
        """ According to ETH's Learn to walk in minutes paper. This is how they compute the 
        reward when using command and actual robot state.
        """
        return np.exp(- np.power(np.linalg.norm(x), 2) / 0.25)

    def compute_heading_reward(self):
        inertial_data = self.robot.get_inertial_data()
        heading_reward = self.phi(self.moving_cmd[2] - inertial_data["angular_velocity"][2])
        return heading_reward

    def compute_forward_reward(self):
        inertial_data = self.robot.get_inertial_data()
        robot_base_heading = inertial_data["rotation"][2]
        forward_velocity = inertial_data["linear_velocity"][0] * np.cos(robot_base_heading) \
            + inertial_data["linear_velocity"][1] * np.sin(robot_base_heading)
        sliding_velocity = - inertial_data["linear_velocity"][0] * np.sin(robot_base_heading) \
            + inertial_data["linear_velocity"][1] * np.cos(robot_base_heading)
        return self.phi(self.moving_cmd[:2] - np.array([forward_velocity, sliding_velocity]))

    def compute_sliding_reward(self):
        inertial_data = self.robot.get_inertial_data()
        return - inertial_data["linear_velocity"][2] ** 2
        
