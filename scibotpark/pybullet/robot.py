from abc import abstractmethod
import os
import numpy as np

import pybullet
import pybullet_data as pb_data
from pybullet_utils import bullet_client

class PybulletRobot:
    def __init__(self,
            default_base_transform= None, # 3-translation + 4-orientation
            pb_control_mode= pybullet.POSITION_CONTROL,
            pb_control_kwargs= dict(), # NOTE This is an advanced argument, you need to know what valid_joint_ids is in your robot.
            pb_client= None,
        ):
        self.default_base_transform = np.array([0,0,0,0,0,0,1]) if default_base_transform is None else default_base_transform
        self.pb_control_kwargs = pb_control_kwargs
        try:
            self.pb_control_mode = getattr(pybullet, pb_control_mode) if isinstance(pb_control_mode, str) else pb_control_mode
        except AttributeError:
            self.pb_control_mode = pb_control_mode
        self.pb_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT) if pb_client is None else pb_client

        self.build_robot_model()

    @abstractmethod
    def load_robot_model(self):
        self._body_id = -1
        raise NotImplementedError("load_robot_model is not implemented")
    def set_robot_dynamics(self):
        pass
    def set_default_joint_states(self):
        pass
    def reset_joint_states(self):
        pass
    
    def build_robot_model(self):
        """ initialization sequence to build and set the robot model """
        self.load_robot_model() # must implement in subclass
        self.set_valid_joint_ids(getattr(self, "valid_joint_types", None))
        self.set_default_joint_states()
        self.set_robot_dynamics()
        self.reset_joint_states()

    def reset(self, base_transform= None):
        """ reset the robot model to initial state
        Args:
            base_transform: a np array of base translation 3-translation + 4 orientation
        """
        if base_transform is None:
            base_transform = self.default_base_transform
        self.pb_client.resetBasePositionAndOrientation(
            self.body_id,
            base_transform[:3],
            base_transform[3:]
        )
        self.pb_client.resetBaseVelocity(self.body_id, [0, 0, 0], [0, 0, 0])
        self.reset_joint_states()
    
    def set_valid_joint_ids(self,
            valid_joint_types= None, # a list of valid joint type, check pybullet docs
        ):
        """ set the valid joint ids for the current robot, and PybulletRobot interface will only
        expose the data of valid joints by default.
        New attributes:
            valid_joint_ids: a nparray of valid joint ids
        """
        if valid_joint_types is None:
            self.valid_joint_ids = np.array(range(self.pb_client.getNumJoints(self._body_id)))
        else:
            self.valid_joint_ids = [
                joint_id for joint_id in range(self.pb_client.getNumJoints(self._body_id)) \
                if self.pb_client.getJointInfo(self.body_id, joint_id)[2] in valid_joint_types
            ]
        self.all_joint_ids = np.array(range(self.pb_client.getNumJoints(self._body_id)))
    
    def get_joint_limits(self,
            modal= "position", # "position", "velocity", "torque"
            valid_joint_only= True,
        ):
        """ get joint limits of current robot (under joint validity configuration)
        Returns:
            limits: a np array of joint limits for the actual robot command, shape (2, n_valid_joints)
        """
        limits = []
        for joint_id in (self.valid_joint_ids if valid_joint_only else self.all_joint_ids):
            joint_info = self.pb_client.getJointInfo(self.body_id, joint_id)
            joint_type = joint_info[2]
            if (joint_type == pybullet.JOINT_PRISMATIC or joint_type == pybullet.JOINT_REVOLUTE)\
                and joint_info[8] == 0. and joint_info[9] == -1.:
                assert modal != "position", "position control for joint {} is not supported".format(joint_info[0])
            if modal == "position":
                limits.append(np.array([joint_info[8], joint_info[9]]))
            elif modal == "velocity":
                limits.append(np.array([-joint_info[11], joint_info[11]]))
            elif modal == "torque":
                limits.append(np.array([-joint_info[10], joint_info[10]]))
            else:
                raise NotImplementedError("modal {} is not implemented".format(modal))
        limits = np.stack(limits, axis=-1) # (2, n_valid_joints)
        return limits

    def get_cmd_limits(self):
        """ return the command limits for the current robot (under joint validity configuration)
        Returns:
            cmd_limits: a np array of joint limits for the actual robot command, shape (2, n_valid_joints)
        """
        if self.pb_control_mode == pybullet.POSITION_CONTROL:
            return self.get_joint_limits(modal= "position")
        elif self.pb_control_mode == pybullet.VELOCITY_CONTROL:
            return self.get_joint_limits(modal= "velocity")
        elif self.pb_control_mode == pybullet.TORQUE_CONTROL:
            return self.get_joint_limits(modal= "torque")
        elif self.pb_control_mode == pybullet.STABLE_PD_CONTROL:
            return self.get_joint_limits(modal= "position")
        else:
            raise ValueError("pb_control_mode {} is not implemented".format(self.pb_control_mode))

    def get_joint_states(self,
            modal= "position", # "position", "velocity", "torque"
            valid_joint_only= True,
        ):
        """ get joint state of current robot (under joint validity configuration)
        Returns:
            joint_states: a np array of joint position/velocity/torque
        """
        joint_states = self.pb_client.getJointStates(self.body_id, (self.valid_joint_ids if valid_joint_only else self.all_joint_ids))
        if modal == "position":
            return np.array([joint_state[0] for joint_state in joint_states], dtype= np.float32)
        elif modal == "velocity":
            return np.array([joint_state[1] for joint_state in joint_states], dtype= np.float32)
        elif modal == "torque":
            return np.array([joint_state[3] for joint_state in joint_states], dtype= np.float32)
        else:
            raise NotImplementedError("modal {} is not implemented".format(modal))

    def send_joints_cmd(self, cmd):
        """ NOTE: This method only sends command to valid joints
        """
        assert len(cmd) == len(self.valid_joint_ids), "cmd length {} is not equal to valid joint ids length {}".format(len(cmd), len(self.valid_joint_ids))
        
        # convert cmd to control_arguments
        if self.pb_control_mode == pybullet.POSITION_CONTROL:
            control_kwargs = dict(targetPositions= cmd,)
        elif self.pb_control_mode == pybullet.VELOCITY_CONTROL:
            control_kwargs = dict(targetVelocities= cmd,)
        elif self.pb_control_mode == pybullet.TORQUE_CONTROL:
            control_kwargs = dict(forces= cmd,)
        elif self.pb_control_mode == pybullet.STABLE_PD_CONTROL:
            control_kwargs = dict(targetPositions= [[c] for c in cmd],)
        else:
            control_kwargs = dict()
        
        # send command to valid joints
        if self.pb_control_mode in [pybullet.POSITION_CONTROL, pybullet.VELOCITY_CONTROL, pybullet.TORQUE_CONTROL]:
            self.pb_client.setJointMotorControlArray(
                self.body_id,
                self.valid_joint_ids,
                controlMode= self.pb_control_mode,
                **control_kwargs,
                **self.pb_control_kwargs,
            )
        elif self.pb_control_mode in [pybullet.STABLE_PD_CONTROL]:
            self.pb_client.setJointMotorControlMultiDofArray(
                self.body_id,
                self.valid_joint_ids,
                controlMode= self.pb_control_mode,
                **control_kwargs,
                **self.pb_control_kwargs,
            )
    
    @abstractmethod
    def get_inertial_data(self, *args, **kwargs):
        """
        Return:
            dict:
                position: (3,)
                linear_velocity: (3,)
                rotation: (3,)
                angular_velocity: (3,)
        """
        return dict(
            position= None,
            linear_velocity= None,
            rotation= None,
            angular_velocity= None,
        )

    def __del__(self):
        """ Incase the robot multibody in pybullet is not desctoyed. """
        try:
            self.pb_client.removeBody(self.body_id)
        except:
            pass

    # must include the following property
    @property
    def body_id(self):
        return self._body_id

class DeltaPositionControlMixin:
    def __init__(self,
            *args,
            delta_control_scale= 1.,
            pb_control_mode= "DELTA_POSITION_CONTROL",
            **kwargs
        ):
        if pb_control_mode == "DELTA_POSITION_CONTROL":
            super().__init__(
                *args,
                pb_control_mode= "POSITION_CONTROL",
                **kwargs,
            )
            self.delta_control_scale = delta_control_scale
        else:
            super().__init__(*args, pb_control_mode= pb_control_mode, **kwargs)

    def get_cmd_limits(self):
        """ return the command limits for the current robot (under joint validity configuration)
        Returns:
            cmd_limits: a np array of joint limits for the actual robot command, shape (2, n_valid_joints)
        """
        if hasattr(self, "delta_control_scale"):
            num_valid_joints = len(self.valid_joint_ids)
            if isinstance(self.delta_control_scale, list):
                assert len(self.delta_control_scale) == num_valid_joints, "Invalid delta_control_scale of length: {}".format(len(self.delta_control_scale))
            limits = np.ones((num_valid_joints,), dtype= np.float32)
            return np.stack([-limits, limits], axis= 0)
        else:
            return super().get_cmd_limits()

    def send_joints_cmd(self, cmd):
        if hasattr(self, "delta_control_scale"):
            current_joint_states = self.get_joint_states()
            current_joint_states += cmd * self.delta_control_scale
            super().send_joints_cmd(current_joint_states)
        else:
            super().send_joints_cmd(cmd)
    