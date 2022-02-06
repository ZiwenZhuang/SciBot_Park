""" some configurations for robot peripherals
"""
import numpy as np
import collections

import pybullet as p

unitree_camera_positions = dict(
    a1= dict(
        front= [0.27, 0, 0],
        back= [-0.2, 0, 0],
        left= [0, 0.1, 0],
        right= [0, -0.1, 0],
        up= [0, 0, 0.07],
        down= [0, 0, -0.07],
    ),
    aliengo= dict(
        front= [0.35, 0, 0],
        back= [-0.33, 0, 0],
        left= [0, 0.1, 0],
        right= [0, -0.1, 0],
        up= [0, 0, 0.06],
        down= [0, 0, -0.06],
    ),
    laikago= dict(
        front= [0.3, 0, 0],
        back= [-0.3, 0, 0],
        left= [0, 0.1, 0],
        right= [0, -0.1, 0],
        up= [0, 0, 0.1],
        down= [0, 0, -0.12],
    )
)
def get_default_camera_dict():
    return dict( # where z-axis is up direction, x-axis is right direction
        front= p.getQuaternionFromEuler([0, 0, -np.pi/2]),
        back= p.getQuaternionFromEuler([0, 0, np.pi/2]),
        left= p.getQuaternionFromEuler([0, 0, 0]),
        right= p.getQuaternionFromEuler([0, 0, np.pi]),
        up= p.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2]),
        down= p.getQuaternionFromEuler([-np.pi/2, 0, -np.pi/2]),
    )
unitree_camera_orientations = collections.defaultdict(get_default_camera_dict)
