""" A importable unitree robot model.
"""
import os
import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

current_file_path = os.path.dirname(os.path.realpath(__file__))

from scibotpark.pybullet.robot import PybulletRobot, DeltaPositionControlMixin
from scibotpark.quadruped_robot.unitree.peripherals import unitree_camera_orientations, unitree_camera_positions

class UniTreeRobot(DeltaPositionControlMixin, PybulletRobot):
    """ The general interface to build UniTree robot into the environment.
    In theory, you can create as many you want in your pybullet client.
    """
    def __init__(self,
            robot_type= "a1",
            camera_fov_kwargs= dict(),
            default_joint_positions= None,
            simulate_timestep= 1./500,
            bullet_debug= False, # if true, will addUserDebugParameter
            **pb_kwargs,
        ):
        self.robot_type = robot_type
        self.camera_fov_kwargs = dict(
            fov= 60,
            nearVal= 0.01,
            farVal= 10,
        ); self.camera_fov_kwargs.update(camera_fov_kwargs)
        self.default_joint_positions = [
            0, 0.6, -1.9,
            0, 0.6, -1.9,
            0, 1.2, -2.0,
            0, 1.2, -2.0,
        ] if default_joint_positions is None else default_joint_positions
        self.bullet_debug = bullet_debug
        self.valid_joint_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
        
        super().__init__(**pb_kwargs)
        self.pb_client.setTimeStep(simulate_timestep)

    def build_robot_model(self):
        super().build_robot_model()
        self.set_onboard_camera()

    def load_robot_model(self):
        if self.robot_type == "a1":
            urdf_file = os.path.join(current_file_path, "data/a1/urdf/a1.urdf")
        elif self.robot_type == "aliengo":
            urdf_file = os.path.join(current_file_path, "data/aliengo/urdf/aliengo.urdf")
        elif self.robot_type == "laikago":
            urdf_file = os.path.join(current_file_path, "data/laikago/urdf/laikago.urdf")
        elif self.robot_type == "go1":
            urdf_file = os.path.join(current_file_path, "data/go1/urdf/go1.urdf")
        else:
            raise ValueError("Invalid robot_type: {}".format(self.robot_type))
        self._body_id = self.pb_client.loadURDF(
            urdf_file,
            self.default_base_transform[:3],
            self.default_base_transform[3:],
            flags= p.URDF_USE_SELF_COLLISION,
            useFixedBase= False
        )

    def set_default_joint_states(self):
        # record the initial joint states into self.default_joint_states
        self.default_joint_states = []
        self.joint_debug_ids = []
        joint_limits = self.get_cmd_limits()
        for idx, j in enumerate(self.valid_joint_ids):
            self.default_joint_states.append(self.default_joint_positions[idx])
            if self.bullet_debug:
                self.joint_debug_ids.append(self.pb_client.addUserDebugParameter(
                    str(self.pb_client.getJointInfo(self.body_id,j)[1]),
                    joint_limits[0, idx],
                    joint_limits[1, idx],
                    self.default_joint_positions[idx]
                ))

    def reset_joint_states(self):
        for idx, j in enumerate(self.valid_joint_ids):
            self.pb_client.resetJointState(self.body_id,j,self.default_joint_states[idx])

    def reset(self, base_transform=None):
        return_ = super().reset(base_transform)
        self.sync_camera_pose()
        return return_

    def sync_camera_pose(self):
        robot_base_positionandorientation = self.pb_client.getBasePositionAndOrientation(self.body_id)
        for camera_name in ["front", "back", "left", "right", "up", "down"]:
            camera_reset_positionandorientation = self.pb_client.multiplyTransforms(
                robot_base_positionandorientation[0],
                robot_base_positionandorientation[1],
                unitree_camera_positions[self.robot_type][camera_name],
                unitree_camera_orientations[self.robot_type][camera_name],
            )
            self.pb_client.resetBasePositionAndOrientation(
                self.camera_ids[camera_name],
                camera_reset_positionandorientation[0],
                camera_reset_positionandorientation[1],
            )

    def set_robot_dynamics(self):
        # enable collision between lower legs
        lower_legs = [2,5,8,11]
        for l0 in lower_legs:
            for l1 in lower_legs:
                if (l1>l0):
                    enableCollision = 1
                    self.pb_client.setCollisionFilterPair(
                        self.body_id,
                        self.body_id,
                        l0,
                        l1,
                        enableCollision
                    )
        for j in range (p.getNumJoints(self.body_id)):
            self.pb_client.changeDynamics(self.body_id,j,linearDamping=0, angularDamping=0)

    def set_onboard_camera(self):
        self.camera_ids = dict()
        for camera_name in ["front", "back", "left", "right", "up", "down"]:
            multi_body_kwargs = dict(
                baseMass= 0.001,
                baseCollisionShapeIndex=self.pb_client.createCollisionShape(
                    shapeType= p.GEOM_BOX,
                    halfExtents= [0.0001, 0.0001, 0.0001],
                ),
            )
            if self.pb_client.getConnectionInfo()['connectionMethod'] == p.GUI:
                multi_body_kwargs.update(dict(
                    baseVisualShapeIndex=self.pb_client.createVisualShape(
                        shapeType= p.GEOM_BOX,
                        halfExtents= [0.005, 0.02, 0.01],
                        rgbaColor= [1, 1, 0.4, 1],
                    ),
                ))
            self.camera_ids[camera_name] = self.pb_client.createMultiBody(**multi_body_kwargs)
            self.pb_client.createConstraint(
                self.body_id,
                -1,
                self.camera_ids[camera_name],
                -1,
                jointType= p.JOINT_FIXED,
                jointAxis= [0, 0, 1],
                parentFramePosition= unitree_camera_positions[self.robot_type][camera_name],
                childFramePosition= [0, 0, 0],
                parentFrameOrientation= unitree_camera_orientations[self.robot_type][camera_name],
                childFrameOrientation= [0, 0, 0, 1],
            )
        self.sync_camera_pose()

    def send_cmd_from_bullet_debug(self):
        assert self.pb_control_mode == p.POSITION_CONTROL, "Not Implemented Error"
        assert self.bullet_debug, "This is only appled in debug mode"
        cmd = []
        for j in self.joint_debug_ids:
            cmd.append(self.pb_client.readUserDebugParameter(j))
        self.send_joints_cmd(cmd)

    def get_onboard_camera_image(self,
            resolution= (480, 480),
            camera_name= "front",
            modal= "rgb", # "rgb", "depth", "rgbd", "segmentation"
        ):
        camera_states = self.pb_client.getBasePositionAndOrientation(self.camera_ids[camera_name])
        camera_target_position = p.multiplyTransforms(
            camera_states[0],
            camera_states[1],
            [0, 0.001, 0],
            [0, 0, 0, 1],
        )[0]
        camera_up_vector = p.multiplyTransforms(
            [0, 0, 0],
            camera_states[1],
            [0, 0, 0.001],
            [0, 0, 0, 1],
        )[0]
        image_data = self.pb_client.getCameraImage(
            width= resolution[0],
            height= resolution[1],
            viewMatrix= self.pb_client.computeViewMatrix(
                cameraEyePosition= camera_states[0],
                cameraTargetPosition= camera_target_position,
                cameraUpVector= camera_up_vector,
            ),
            projectionMatrix= self.pb_client.computeProjectionMatrixFOV(
                aspect= resolution[0]/resolution[1],
                **self.camera_fov_kwargs
            ),
        )

        rgb_image = image_data[2].reshape(resolution + (4,)).astype(np.float32)[..., :3] / 255.
        if modal == "rgb":
            return rgb_image
        elif modal == "depth":
            return image_data[3].reshape(resolution + (1,))
        elif modal == "rgbd":
            return np.concatenate([
                rgb_image,
                image_data[3].reshape(resolution + (1,)),
            ], axis= -1)
        elif modal == "segmentation":
            return image_data[4].reshape(resolution + (1,))
        else:
            raise NotImplementedError("Not implemented modal {}".format(modal))

    def get_inertial_data(self):
        """
        Return:
            dict:
                position: (3,)
                linear_velocity: (3,)
                rotation: (3,)
                angular_velocity: (3,)
        """
        if not hasattr(self, "imu_link_idx"):
            self.imu_link_idx = -1
            for idx in range(self.pb_client.getNumJoints(self.body_id)-1, -1):
                joint_info = self.pb_client.getJointInfo(self.body_id, idx)
                if "imu" in joint_info[1]:
                    self.imu_link_idx = idx
                    break
            # if self.imu_link_idx == -1: No imu joint found using base link

        if self.imu_link_idx == -1:
            position, orientation = self.pb_client.getBasePositionAndOrientation(self.body_id)
            linear_velocity, angular_velocity = self.pb_client.getBaseVelocity(self.body_id)
        else:
            state = self.pb_client.getLinkState(self.body_id, self.imu_link_idx, computeForwardKinematics= True)
            position = state[4]
            orientation = state[5]
            linear_velocity = state[6]
            angular_velocity = state[7]
        return dict(
            position= np.array(position),
            linear_velocity= np.array(linear_velocity),
            rotation= np.array(p.getEulerFromQuaternion(orientation)),
            angular_velocity= np.array(angular_velocity),
        )
