import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

class RobotBasePerturbationMixin:
    """ Mixin implementation for adopting random external force to the robot base.
    """
    def __init__(self,
            *args,
            external_max_forces= (5, 5, 0.2), # x, y, z axis
            perturbation_position_range= (0.1, 0.05, 0.02), # half-size of the force exertion point range
            perturbation_prob= 0.2, # the probability of adding it at each timestep
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.external_max_forces = external_max_forces if isinstance(external_max_forces, np.ndarray) else np.array(external_max_forces)
        self.perturbation_position_range = perturbation_position_range if isinstance(perturbation_position_range, np.ndarray) else np.array(perturbation_position_range)
        self.perturbation_prob = perturbation_prob

    def step_simulation_from_action(self, action):
        if np.random.uniform() < self.perturbation_prob:
            force = np.random.uniform(
                low= -self.external_max_forces,
                high= self.external_max_forces,
            )
            position = np.random.uniform(
                low= -self.perturbation_position_range,
                high= self.perturbation_position_range,
            )
            self.pb_client.applyExternalForce(
                self.robot.body_id,
                -1, # base link id
                forceObj= force,
                posObj= position,
                flags= p.LINK_FRAME,
            )
        super().step_simulation_from_action(action)
