import os
import numpy as np
import gym
import pybullet
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from scibotpark.pybullet.robot import PybulletRobot

file_directory = os.path.dirname(os.path.realpath(__file__))
class ShadowHandRobot(PybulletRobot):
    def __init__(self,
            init_forearm_position= [0, -0.37, 0.1], # absolute value
            init_forearm_orientation_pry= [1.5714,0.0,3.14159], # relative to world frame
            init_joints_positions= [0] * 5 + [-0.69] + [0] * 22, # including disabled joint positions
            enable_forearm_rotation= False, # if False, action space will be 1 dim less
            enable_forearm_translation= False, # if False, action space will be 3 dim less, recommend to set False when arm base is not fixed
            forearm_translation_by_delta_scale= 0.1,
            normalize_action_space= False,
            robot_dynamics_kwargs= dict(),
            **pb_kwargs,
        ):
        self.init_forearm_position = np.array(init_forearm_position)
        self.init_forearm_orientation_pry = np.array(init_forearm_orientation_pry)
        self.init_joints_positions = np.array(init_joints_positions)
        self.enable_forearm_rotation = enable_forearm_rotation
        self.enable_forearm_translation = enable_forearm_translation
        self.forearm_translation_by_delta_scale = forearm_translation_by_delta_scale
        self.normalize_action_space = normalize_action_space
        self.robot_dynamics_kwargs = dict(); self.robot_dynamics_kwargs.update(robot_dynamics_kwargs)

        super().__init__(**pb_kwargs)

    def load_robot_model(self):
        self._body_id = self.pb_client.loadURDF(
            os.path.join(file_directory, "shadow_hand_ign", "urdf", "shadow_hand.urdf"),
            basePosition= self.init_forearm_position,
            baseOrientation= pybullet.getQuaternionFromEuler(self.init_forearm_orientation_pry),
            useFixedBase= True,
            flags= 0 \
                | pybullet.URDF_USE_SELF_COLLISION \
                # | pybullet.URDF_USE_SELF_COLLISION_INCLUDE_PARENT \
                | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS \
                # | pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL \
                # | pybullet.URDF_INITIALIZE_SAT_FEATURES \
                # | pybullet.URDF_ENABLE_CACHED_GRAPHICS_SHAPES \
                | pybullet.URDF_MAINTAIN_LINK_ORDER \
                | pybullet.URDF_USE_INERTIA_FROM_FILE \
                | 0
        )

    def set_robot_dynamics(self):
        all_hand_joint_limits = self.get_joint_limits(valid_joint_only= False)
        all_hand_num_joints = all_hand_joint_limits.shape[1]
        for j_id in range(all_hand_num_joints):
            joint_info = self.pb_client.getJointInfo(self.body_id, j_id)
            if joint_info[1] == b"wrist_joint":
                wrist_joint_id = joint_info[0]
                break
        for joint_id in range(-1, all_hand_num_joints):
            self.pb_client.changeDynamics(self.body_id, joint_id,
                jointLowerLimit= all_hand_joint_limits[0, joint_id],
                jointUpperLimit= all_hand_joint_limits[1, joint_id],
                **self.robot_dynamics_kwargs
            )
            if joint_id > wrist_joint_id:
                joint_name = self.pb_client.getJointInfo(self.body_id, joint_id)[1].decode("utf-8")
                if ("middle" in joint_name or "distal" in joint_name):
                    self.pb_client.setCollisionFilterPair(
                        self.body_id, self.body_id,
                        joint_id, wrist_joint_id,
                        enableCollision= True,
                    )
        assert all_hand_num_joints == len(self.init_joints_positions), \
            "Invalid init_joint_positions, should be len: {}, got len: {}".format(self.hand_num_joints, len(self.init_joints_positions))
        
    def set_valid_joint_ids(self, valid_joint_types=None):
        super().set_valid_joint_ids(valid_joint_types)
        valid_joint_ids_mask = np.ones(len(self.valid_joint_ids))
        if not self.enable_forearm_rotation:
            self.enabled_joint_idx_mask[3] = False
        if not self.enable_forearm_translation:
            self.enabled_joint_idx_mask[:3] = False
        self.valid_joint_ids = self.valid_joint_ids[valid_joint_ids_mask.astype(bool)]

    def set_default_joint_states(self):
        self.default_joint_states = self.init_joints_positions
    
    def reset_joint_states(self):
        # self.pb_client.resetBasePositionAndOrientation(
        #     bodyUniqueId= self.robot.body_id,
        #     posObj= self.init_forearm_position,
        #     ornObj= pybullet.getQuaternionFromEuler(self.init_forearm_orientation_pry),
        # )
        for joint_id, joint_position in enumerate(self.default_joint_states):
            self.pb_client.resetJointState(
                bodyUniqueId= self.body_id,
                jointIndex= joint_id,
                targetValue= joint_position,
            )

    def get_cmd_limits(self):
        cmd_limits = super().get_cmd_limits()
        if self.normalize_action_space:
            cmd_limits = np.ones(cmd_limits.shape, dtype= np.float32)
            cmd_limits[0] *= -1
        return cmd_limits

    def send_joints_cmd(self, cmd):
        if self.normalize_action_space:
            cmd_limits = super().get_cmd_limits()
            cmd = (cmd + 1) * (cmd_limits[1] + cmd_limits[0]) / 2
        if self.enable_forearm_translation and self.forearm_translation_by_delta_scale and self.pb_control_mode == pybullet.POSITION_CONTROL:
            forearm_translation = self.get_joint_states()[:3]
            cmd[:3] *= self.forearm_translation_by_delta_scale
            cmd[:3] += forearm_translation
        return super().send_joints_cmd(cmd)

    def get_reached_positions(self):
        """ I didn't put this method in base robot implementation, because it is not well defined. 
        It is hard to say which positions in the robot link should be used as 'reached'.
        """
        link_states = self.pb_client.getLinkStates(self.body_id, self.all_joint_ids)
        return np.array([i[0] for i in link_states])