import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from scibotpark.utils.quick_args import save__init__args

class MetaQuadruped_ConfigSampler:
    def __init__(self,
            base_mass_range= (5.0, 20.0), # kg
            base_length_range= (0.05, 0.4), # meter, x axis
            base_width_range= (0.02, 0.08), # meter, y axis
            base_height_range= (0.02, 0.1), # meter, z axis
            hip_radius_range= (0.006, 0.04), # meter,
            thigh_mass_range= (0.1, 2), # kg
            thigh_length_range= (0.1, 0.3), # meter,
            shin_mass_range= (0.1, 2), # kg
            shin_length_range= (0.12, 0.3), # meter,
            foot_mass_range= (0.01, 0.2), # kg
            foot_size_range= (0.01, 0.02), # meter,
            motor_torque_range= (25, 40), # Nm
        ):
        save__init__args(locals())
    
    def sample(self, robot_kwargs):
        robot_kwargs.update(dict(
            base_mass= np.random.uniform(*self.base_mass_range),
            base_size= (
                np.random.uniform(*self.base_length_range),
                np.random.uniform(*self.base_width_range),
                np.random.uniform(*self.base_height_range),
            ),
            hip_size= (
                np.random.uniform(*self.hip_radius_range),
                0.03,
            ),
            thigh_mass= np.random.uniform(*self.thigh_mass_range),
            thigh_size= (
                0.01,
                np.random.uniform(*self.thigh_length_range),
            ),
            shin_mass= np.random.uniform(*self.shin_mass_range),
            shin_size= (
                0.01,
                np.random.uniform(*self.shin_length_range),
            ),
            foot_mass= np.random.uniform(*self.foot_mass_range),
            foot_size= (np.random.uniform(*self.foot_size_range)),
        ))
        pb_control_kwargs = dict(
            forces= [np.random.uniform(*self.motor_torque_range) for _ in range(12)],
        )
        if "pb_control_kwargs" in robot_kwargs:
            robot_kwargs["pb_control_kwargs"].update(pb_control_kwargs)
        else:
            robot_kwargs["pb_control_kwargs"] = pb_control_kwargs
        
        return robot_kwargs