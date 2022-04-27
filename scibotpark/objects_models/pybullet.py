import numpy as np
import os
from os import path as osp

import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

current_file_path = osp.abspath(__file__)

def get_ycb_file_path(object_name: str):
    """ Return the abspath of the urdf file given the object name (be aware of the uppercase)
    NOTE: the name has to start with 'Ycb'
    """
    assert object_name.startswith("Ycb")
    return osp.join(osp.dirname(current_file_path), "ycb_objects", object_name, "model.urdf")

class MultiObjectsMixin:
    """ A mixin class implementing common methods of loading objects and resetting them.
    Due to different robot have different observation_space and action_space, the workable env
    should be found in their corresponding directory
    """
    def __init__(self,
            object_names: list, # a list of objects to put in the scene (allows duplicate)
            position_sample_region: np.ndarray= None, # shape (2, 3)
            **kwargs,
        ):
        self.object_names = object_names
        # representing where to spawn the objects randomly, see default example
        self.position_sample_region = np.array([
            [0.2, -0.25, 0.1],
            [0.6, 0.25, 0.3],
        ]) if position_sample_region is None else position_sample_region
        super().__init__(**kwargs)

    def _build_surroundings(self):
        super()._build_surroundings()
        self._object_body_ids = []
        for name in self.object_names:
            if name.startswith("Ycb"):
                file_path = get_ycb_file_path(name)
            else:
                raise NotImplementedError
            
            # load model
            object_position = np.random.uniform(self.position_sample_region[0], self.position_sample_region[1])
            object_xyz_euler = np.random.uniform(np.array([-np.pi]*3), np.array([np.pi]*3))
            if file_path.endswith(".urdf"):
                self._object_body_ids.append(self.pb_client.loadURDF(
                    file_path,
                    basePosition= object_position,
                    baseOrientation= p.getQuaternionFromEuler(object_xyz_euler),
                ))
            else:
                raise NotImplementedError
            
    def _reset_surroundings(self, *args, **kwargs):
        for body_id in self._object_body_ids:
            object_position = np.random.uniform(self.position_sample_region[0], self.position_sample_region[1])
            object_xyz_euler = np.random.uniform(np.array([-np.pi]*3), np.array([np.pi]*3))
            self.pb_client.resetBasePositionAndOrientation(
                body_id,
                object_position,
                p.getQuaternionFromEuler(object_xyz_euler),
            )
        
            
