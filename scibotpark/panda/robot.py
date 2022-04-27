import numpy as np
import os
from os import path as osp

import pybullet as p
import pybullet_data as pb_data

from scibotpark.pybullet.robot import PybulletRobot

class PandaRobot(PybulletRobot):
    def __init__(self,
            effector_rest_euler= [np.pi, 0., 0.],
            arm_control_mode= "translation",
            movement_stepsize= 0.1, # for length when cartesian movement of the end effector
            rotation_stepsize= 10, # for maximum rotation delta when cartesian movement of the end effector
            gripper_max_force= 10., # the force send to gripper PID controller
            **pb_kwargs,
        ):
        """
        arm_control_mode:
            "translation": The gripper only face down, move towards x,y,z axis
            "trans_yaw": The gripper face down, move towardsd x,y,z axis and rotate around z axis
            "cartesian": The gripper moves all 6 DoF
            "joint": The robot is controlled directly by joint motor commands
        """
        self.effector_rest_euler = effector_rest_euler
        self.arm_control_mode = arm_control_mode
        self.movement_stepsize = movement_stepsize
        self.rotation_stepsize = rotation_stepsize
        self.gripper_max_force = gripper_max_force
        self.local_workspace_box = np.array([
            [0.08, -0.8, 0.0],
            [0.8, 0.8, 0.4],
        ]) # relative to base frame
        self.initial_position_space = np.array([
            [0.1, -0.5, 0.2],
            [0.5, 0.5, 0.4],
        ]) # relative to base frame
        self._get_finger_position_from_cmd = lambda x: (x+1) / 50. # from [-1, 1] to [0, 2/50]
        self.valid_joint_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]

        if not self.arm_control_mode == "joint":
            pb_kwargs["pb_control_mode"] = p.POSITION_CONTROL

        super().__init__(**pb_kwargs)

    def load_robot_model(self):
        urdf_file = osp.join(pb_data.getDataPath(), "franka_panda", "panda.urdf")
        assert osp.exists(urdf_file), f"{urdf_file} not exist"
        self._body_id = self.pb_client.loadURDF(
                urdf_file,
                basePosition= self.default_base_transform[:3],
                baseOrientation= self.default_base_transform[3:],
                useFixedBase= True,
            )
        self.end_effector_joint_id = 8
        # set relative position of the Point of Interest of the end effector, which is the
        # coordinate in the end effector frame
        self.poi_offset = np.array([0, 0, 0.1], dtype= np.float32)

        # set cmd visualization body, it has only visual shape so that it will not effect
        # the physical simulation. If you want to see it, please set its alpha value. 
        self._poi_visualization_body_id = self.pb_client.createMultiBody(
            baseVisualShapeIndex= self.pb_client.createVisualShape(
                p.GEOM_SPHERE,
                radius= 0.02,
                rgbaColor= [1., 0., 1., 0.],
            )
        )
        self._cmd_visualization_body_id = self.pb_client.createMultiBody(
            baseVisualShapeIndex= self.pb_client.createVisualShape(
                p.GEOM_SPHERE,
                radius= 0.02,
                rgbaColor= [0., 1., 0.5, 0.],
            )
        )

    def set_valid_joint_ids(self, valid_joint_types=None):
        return_ = super().set_valid_joint_ids(valid_joint_types)
        
        # set valid joint limits (for calculate inverse kinematics)
        joints_limits = []
        for joint_id in self.valid_joint_ids:
            if joint_id <= self.end_effector_joint_id:
                joints_limits.append(self.pb_client.getJointInfo(self.body_id, joint_id)[8:10])
        self.joints_limits = np.array(joints_limits).transpose() # (2, n_valid_joints)
        self.joints_ranges = self.joints_limits[1] - self.joints_limits[0]

        return return_

    def set_default_joint_states(self):
        self.effector_rest_orientation = self.pb_client.getQuaternionFromEuler(self.effector_rest_euler)
        self.joints_rest_poses = [0.0, 0.09, -0.41, -2.11, 0.16, 2.17, .43, 0, 0]
        
    def _get_joints_by_poi_transform(self, poi_transform, add_limits= True):
        """ poi: point of interest (on the end effector)
        """
        position_offset, _ = p.multiplyTransforms(
            [0, 0, 0],
            poi_transform[3:],
            *p.invertTransform(
                self.poi_offset,
                np.array([0,0,0,1])
            ),
        )
        """ Compute the joint positions given robot point of interest transform. """
        if add_limits:
            joint_positions = self.pb_client.calculateInverseKinematics(
                self.body_id, self.end_effector_joint_id,
                poi_transform[:3] + position_offset,
                poi_transform[3:],
                lowerLimits= self.joints_limits[0],
                upperLimits= self.joints_limits[1],
                jointRanges= self.joints_ranges,
                restPoses= self.joints_rest_poses,
            )
        else:
            joint_positions = self.pb_client.calculateInverseKinematics(
                self.body_id, self.end_effector_joint_id,
                poi_transform[:3] + position_offset,
                poi_transform[3:],
            )
        return joint_positions

    def get_poi_world_transform(self):
        """ return the world transform of robot point of interest """
        effector_state = self.pb_client.getLinkState(self.body_id, self.end_effector_joint_id)
        poi_position, poi_orientation = p.multiplyTransforms(
            effector_state[4],
            effector_state[5],
            self.poi_offset,
            np.array([0,0,0,1]),
        )
        return np.array(poi_position + poi_orientation) # shape (7,) of position and quaternion orientation

    def get_finger_state(self):
        """ return a scalar from -1 to 1 telling whether the finger on the effector is opened. """
        joint0_state = self.pb_client.getJointState(self.body_id, self.end_effector_joint_id+1)
        joint1_state = self.pb_client.getJointState(self.body_id, self.end_effector_joint_id+2)
        return (joint0_state[0] + joint1_state[0]) * 50 - 1

    def sample_random_initial_position(self):
        """ return a random initial position in the workspace """
        position = np.random.uniform(
            self.initial_position_space[0], self.initial_position_space[1]
        )
        base_position, base_orientation = self.pb_client.getBasePositionAndOrientation(self.body_id)
        return np.array(p.multiplyTransforms(
            base_position,
            base_orientation,
            position,
            np.array([0,0,0,1]),
        )[0], dtype= np.float32)

    def reset_joint_states(self):
        for joint_id in range(self.end_effector_joint_id+1):
            self.pb_client.resetJointState(self.body_id, joint_id, self.joints_rest_poses[joint_id])
        if self.arm_control_mode == "joint": return

        # reset arm position at random position
        init_position = self.sample_random_initial_position()
        init_poi_transform = np.concatenate([init_position, self.effector_rest_orientation], axis= 0)
        joint_positions = self._get_joints_by_poi_transform(init_poi_transform)
        for joint_id in range(self.pb_client.getNumJoints(self.body_id)):
            if joint_id < len(joint_positions):
                joint_position = joint_positions[joint_id]
            else:
                np.random.uniform(*self.pb_client.getJointInfo(self.body_id, joint_id)[8:10])
            self.pb_client.resetJointState(self.body_id, joint_id, joint_position)

        # reset _poi_visualization_body position
        poi_transform = self.get_poi_world_transform()
        self.pb_client.resetBasePositionAndOrientation(self._poi_visualization_body_id,
            poi_transform[:3],
            poi_transform[3:],
        )
        
    def get_cmd_limits(self):
        if self.arm_control_mode == "joint":
            return super().get_cmd_limits()
        
        if self.arm_control_mode == "translation":
            limit = np.array([self.movement_stepsize] * 3 + [1])
            # shape (4,) for translation delta and effector open/close state
            return np.stack([-limit, limit], axis= 0).astype(np.float32)
        elif self.arm_control_mode == "cartesian" or self.arm_control_mode == "trans_yaw":
            limit_translation = np.array([self.movement_stepsize] * 3)
            if self.arm_control_mode == "cartesian":
                limit_rotation = np.array([self.rotation_stepsize] * 3) # roll (+x), pitch (+y), yaw (+z)
            else:
                limit_rotation = np.array([self.rotation_stepsize]) # only yaw axis (+z)
            limit = np.concatenate([limit_translation, limit_rotation, np.array([1])], axis= 0)
            # shape (7,) for translation delta, rotation delta and effector open/close state
            return np.stack([-limit, limit], axis= 0).astype(np.float32)

    def send_joints_cmd(self, cmd):
        if self.arm_control_mode == "joint":
            return super().send_joints_cmd(cmd)

        # cartesian control or translation control
        poi_world_transform = self.get_poi_world_transform()
        target_poi_position = poi_world_transform[:3] + cmd[:3]
        target_poi_position = self.clip_local_space(target_poi_position, self.local_workspace_box)
        if self.arm_control_mode == "cartesian":
            poi_world_euler = self.pb_client.getEulerFromQuaternion(poi_world_transform[3:])
            target_poi_euler = poi_world_euler + cmd[3:-1]
            target_poi_orientation = self.pb_client.getQuaternionFromEuler(target_poi_euler)
        elif self.arm_control_mode == "trans_yaw":
            poi_world_euler = self.pb_client.getEulerFromQuaternion(poi_world_transform[3:])
            target_poi_euler = np.concatenate([
                self.effector_rest_euler[:2],
                poi_world_euler[2:3] + cmd[3:4],
            ])
            target_poi_orientation = self.pb_client.getQuaternionFromEuler(target_poi_euler)
        else:
            target_poi_orientation = self.effector_rest_orientation
        target_poi_transform = np.concatenate([target_poi_position, target_poi_orientation])

        self.pb_client.resetBasePositionAndOrientation(self._poi_visualization_body_id,
            poi_world_transform[:3],
            poi_world_transform[3:],
        )
        self.pb_client.resetBasePositionAndOrientation(self._cmd_visualization_body_id,
            target_poi_position,
            target_poi_orientation,
        )

        joint_positions = self._get_joints_by_poi_transform(target_poi_transform)
        
        self.pb_client.setJointMotorControlArray(self.body_id, 
            list(range(self.end_effector_joint_id+1)),
            p.POSITION_CONTROL,
            targetPositions= joint_positions
        )
        self.pb_client.setJointMotorControl2(self.body_id, 
            self.end_effector_joint_id+1,
            p.POSITION_CONTROL,
            targetPosition= self._get_finger_position_from_cmd(cmd[-1]),
            force= self.gripper_max_force,
        )
        self.pb_client.setJointMotorControl2(self.body_id, 
            self.end_effector_joint_id+2,
            p.POSITION_CONTROL,
            targetPosition= self._get_finger_position_from_cmd(cmd[-1]),
            force= self.gripper_max_force,
        )

    def set_poi_visiblity(self, visible= True):
        """ set the poi body visible or invisible, only for visualization """
        self.pb_client.changeVisualShape(
            self._poi_visualization_body_id,
            -1,
            rgbaColor= [1, 0, 1, (0.5 if visible else 0.)]
        )
        self.pb_client.changeVisualShape(
            self._cmd_visualization_body_id,
            -1,
            rgbaColor= [0, 1, 0.5, (0.5 if visible else 0.)]
        )

    