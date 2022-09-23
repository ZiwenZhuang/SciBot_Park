from abc import abstractmethod
import os
import numpy as np

import pybullet
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from scibotpark.pybullet.robot import PybulletRobot

class PybulletEnv:
    """ Some helper functions that can be used in almost all pybullet env """
    def __init__(self,
            nsubsteps= 1,
            additional_search_path= None,
            pb_object_dynamics_parameter= dict(),
            pb_engine_parameter= dict(),
            pb_client= None
        ):
        """ store parameters and initialize world and robot
        NOTE: In this class, there is not action_space or observation_space concept
        """
        self.nsubsteps = nsubsteps
        self.pb_object_dynamics_parameter = pb_object_dynamics_parameter # change object dynamics in general (e.g. ground plane)
        self.pb_engine_parameter = pb_engine_parameter # used in self.pb_client.setPhysicsEngineParameter
        self.pb_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT) if pb_client is None else pb_client
        if additional_search_path is not None: self.pb_client.setAdditionalSearchPath(additional_search_path)

        self._initialize_world()
        self._build_surroundings()
        self._build_robot()
    
    def _initialize_world(self):
        self.pb_client.setRealTimeSimulation(0)
        self.pb_client.setPhysicsEngineParameter(**self.pb_engine_parameter)
        self.pb_client.setGravity(0, 0, -9.81)
        if hasattr(self, "simulate_timestep") and self.simulate_timestep:
            self.pb_client.setTimeStep(self.simulate_timestep)
        
        self._nsteps_after_reset = 0 # use to set done based on horizon.

    def _build_surroundings(self):
        self.plane_id = self.pb_client.loadURDF(
            os.path.join(pb_data.getDataPath(), "plane.urdf"),
            [0, 0, 0],
            useFixedBase= True,
        )
        self.pb_client.changeDynamics(self.plane_id, -1, **self.pb_object_dynamics_parameter)

    def _build_robot(self):
        pass

    def _reset_surroundings(self, *args, **kwargs):
        pass # no surroundings in the world (in this class)

    def _reset_robot(self, *args, **kwargs):
        if not self.robot is None:
            self.robot.reset(self.robot_kwargs.get("default_base_transform", None))

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError
    
    def step_simulation_from_action(self, action):
        self.robot.send_joints_cmd(action)
        for _ in range(self.nsubsteps): self.pb_client.stepSimulation()
        self._nsteps_after_reset += 1

    def is_done(self, *args, **kwargs):
        return self._nsteps_after_reset >= self.horizon if hasattr(self, "horizon") else False
    
    def reset(self, *args, **kwargs):
        self._nsteps_after_reset = 0 # use to set done based on horizon.
        try:
            self._reset_surroundings(*args, **kwargs)
            self._reset_robot(*args, **kwargs)
        except NotImplementedError as e:
            # This could reduce the memroy leak problem of Pybullet, but slower.
            if hasattr(self, "_robot"):
                # incase the old _robot handler in the system disrrupts the new simulation.
                delattr(self, "_robot")
            self.pb_client.resetSimulation()
            self._initialize_world()
            self._build_surroundings()
            self._build_robot()
        self.pb_client.stepSimulation()
        return self._get_obs()

    # must include the following properties
    @property
    def robot(self) -> PybulletRobot:
        return getattr(self, "_robot", None)
