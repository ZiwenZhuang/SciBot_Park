import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from scibotpark.pybullet.robot import PybulletRobot, DeltaPositionControlMixin

class MetaQuadrupedRobot(DeltaPositionControlMixin, PybulletRobot):
    """ This class implements a robot with 4 legs.
    The robot has 12 joints to control the locomotion.
    The size of the base and legs can be specified in the constructor.
    NOTE: +x is the direction of the front of the robot, +z is the direction of the top of the robot.
    """
    def __init__(self,
            base_mass= 14.0, # kg
            base_size= (0.16, 0.04, 0.03), # half size of (x, y, z) axis, box
            hip_size= (0.01, 0.03), # (radius, height), cylinder
            thigh_mass= 0.1, # kg
            thigh_size= (0.01, 0.1), # (radius, height), capsule
            shin_mass= 0.1, # kg
            shin_size= (0.01, 0.12), # (radius, height), capsule
            foot_mass= 0.01, # kg
            foot_size= 0.012, # radius, sphere
            foot_lateral_friction= 0.5,
            leg_bend_angle= np.pi/4, # rad
            pb_control_kwargs= dict(),
            default_base_transform= [0, 0, 0.3, 0, 0, 0, 1],
            **kwargs,
        ):
        self.configuration= dict(
            base_mass = base_mass,
            base_size = base_size,
            hip_mass = 0.001,
            hip_size = hip_size,
            thigh_mass = thigh_mass,
            thigh_size = thigh_size,
            shin_mass = shin_mass,
            shin_size = shin_size,
            foot_mass = foot_mass,
            foot_size = foot_size,
            foot_lateral_friction = foot_lateral_friction
        )
        self.leg_bend_angle = leg_bend_angle
        self.valid_joint_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
        _pb_control_kwargs = dict(
            forces= [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        ); _pb_control_kwargs.update(pb_control_kwargs)
        self.set_all_joint_position_limits()
        super().__init__(
            pb_control_kwargs= _pb_control_kwargs,
            default_base_transform= default_base_transform,
            **kwargs,
        )

    def load_robot_model(self, configuration= None):
        """ Build the robot model based on configs. Not using urdf.
        """
        if configuration is None:
            configuration = self.configuration
        # Load the shapes
        component_ids = self.get_component_ids(configuration)
        # build and compute relative position
        self.build_multibody(component_ids, configuration)

    def set_all_joint_position_limits(self):
        """ This method only set the attribute, but not change any setting in pybullet simulator. """
        self.all_joint_position_limits = np.ones((2, 24))
        self.all_joint_position_limits[1] *= -1 # invalid joints will not have valid limits
        for joint_id in [0, 12]: # left hip
            self.all_joint_position_limits[:, joint_id] = [-np.pi/6, np.pi/4]
        for joint_id in [6, 18]: # right hip
            self.all_joint_position_limits[:, joint_id] = [-np.pi/4, np.pi/6]
        for joint_id in [1, 7, 13, 19]: # thigh
            self.all_joint_position_limits[:, joint_id] = [self.leg_bend_angle - np.pi, self.leg_bend_angle]
        for joint_id in [3, 9, 15, 21]: # shin
            self.all_joint_position_limits[:, joint_id] = [0, np.pi*8/9]

    def get_joint_limits(self, modal="position", valid_joint_only=True):
        # assert modal == "position", "Only position limits are supported, get {}".format(modal)
        if modal == "position":
            if valid_joint_only:
                return self.all_joint_position_limits[:, self.valid_joint_ids]
            else:
                return self.all_joint_position_limits
        elif modal == "velocity":
            # This limit only provides a shape, the space is not meaningful.
            num_joints = len(self.valid_joint_ids) if valid_joint_only else self.all_joint_position_limits.shape[1]
            limit = np.ones((2, num_joints)) * np.inf
            limit[0] = -np.inf
            return limit
        elif modal == "torque":
            limit = np.array([self.pb_control_kwargs["forces"], self.pb_control_kwargs["forces"]])
            limit[0] *= -1
            if not valid_joint_only:
                num_joints = len(self.valid_joint_ids) if valid_joint_only else self.all_joint_position_limits.shape[1]
                limit_ = np.ones((2, num_joints)) * np.inf
                limit_[0] = -np.inf
                limit_[:, self.valid_joint_ids] = limit
                limit = limit_
            return limit

    def set_robot_dynamics(self):
        for joint_id in self.valid_joint_ids:
            self.pb_client.changeDynamics(self.body_id, joint_id,
                jointLowerLimit=self.all_joint_position_limits[0, joint_id],
                jointUpperLimit=self.all_joint_position_limits[1, joint_id],
            )
        for link_id in [4, 10, 16, 22]: # foot
            self.pb_client.changeDynamics(self.body_id, link_id,
                lateralFriction= self.configuration["foot_lateral_friction"],
            )

    def build_multibody(self, component_ids, configuration):
        directions = np.array([
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ]) # NOTE: this order is important for controled joint indices
        link_masses, link_collision_ids, = [], []
        link_visual_ids, link_positions, = [], []
        link_orientations, link_parent_ids = [], []
        link_joint_types, link_joint_axes = [], []
        for leg_idx, direct in enumerate(directions):
            # 4 legs
            n_link_per_leg = 6
            # hip
            link_masses.append(configuration["hip_mass"])
            link_collision_ids.append(component_ids['hip_collision_id'])
            link_visual_ids.append(component_ids['hip_visual_id'])
            hip_horizontal_shift = direct * configuration["base_size"][:2]
            hip_horizontal_shift[1] += configuration["hip_size"][0] * direct[1]
            link_positions.append(np.concatenate([hip_horizontal_shift, np.array([0])]))
            link_orientations.append(p.getQuaternionFromEuler([0, np.pi/2, 0])) # lie down the cylinder
            link_parent_ids.append(0)
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append(np.array([0, 0, 1]))
            # hip to thigh
            link_masses.append(0.001)
            link_collision_ids.append(component_ids['hip_thigh_collision_id'])
            link_visual_ids.append(component_ids['hip_thigh_visual_id'])
            hip_thigh_shift = direct[1] * (configuration["hip_size"][0] + configuration["thigh_size"][0])
            link_positions.append(np.array([0, hip_thigh_shift, 0]))
            link_orientations.append(p.getQuaternionFromEuler([np.pi/2, 0, 0,])) # point outwards
            link_parent_ids.append(leg_idx * n_link_per_leg + 1)
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append(np.array([0, 0, 1])) # relative to child's z axis
            # thigh
            link_masses.append(configuration["thigh_mass"])
            link_collision_ids.append(component_ids['thigh_collision_id'])
            link_visual_ids.append(component_ids['thigh_visual_id'])
            link_positions.append(np.array([configuration["thigh_size"][1]/2, 0, 0]))
            link_orientations.append(p.getQuaternionFromEuler([0, np.pi/2, 0]))
            link_parent_ids.append(leg_idx * n_link_per_leg + 2)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append(np.array([0, 0, 0]))
            # knee
            link_masses.append(0.001)
            link_collision_ids.append(component_ids['knee_collision_id'])
            link_visual_ids.append(component_ids['knee_visual_id'])
            link_positions.append(np.array([0, 0, configuration["thigh_size"][1]/2 + configuration["shin_size"][0]]))
            link_orientations.append(p.getQuaternionFromEuler([0, -np.pi/2, 0]))
            link_parent_ids.append(leg_idx * n_link_per_leg + 3)
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append(np.array([0, 0, 1])) # relative to child's z axis
            # shin
            link_masses.append(configuration["shin_mass"])
            link_collision_ids.append(component_ids['shin_collision_id'])
            link_visual_ids.append(component_ids['shin_visual_id'])
            link_positions.append(np.array([configuration["shin_size"][1]/2, 0, 0]))
            link_orientations.append(p.getQuaternionFromEuler([0, np.pi/2, 0]))
            link_parent_ids.append(leg_idx * n_link_per_leg + 4)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append(np.array([0, 0, 0]))
            # foot
            link_masses.append(configuration["foot_mass"])
            link_collision_ids.append(component_ids['foot_collision_id'])
            link_visual_ids.append(component_ids['foot_visual_id'])
            link_positions.append(np.array([0, 0, configuration["shin_size"][1]/2 + configuration["foot_size"]]))
            link_orientations.append(p.getQuaternionFromEuler([0, 0, 0]))
            link_parent_ids.append(leg_idx * n_link_per_leg + 5)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append(np.array([0, 0, 0]))
        
        # generate the multibody
        self._body_id = self.pb_client.createMultiBody(
            baseMass=configuration["base_mass"],
            baseCollisionShapeIndex=component_ids['base_collision_id'],
            baseVisualShapeIndex=component_ids['base_visual_id'],
            basePosition=self.default_base_transform[:3],
            baseOrientation=self.default_base_transform[3:],
            baseInertialFramePosition=np.array([0, 0, 0]),
            baseInertialFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_ids,
            linkVisualShapeIndices=link_visual_ids,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0, 0, 0] for _ in range(len(link_masses))],
            linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in range(len(link_masses))],
            linkParentIndices=link_parent_ids,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            flags= p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )

    def get_component_ids(self, configuration):
        component_ids = dict(
            base_collision_id = self.pb_client.createCollisionShape(p.GEOM_BOX,
                halfExtents=configuration["base_size"],
            ),
            base_visual_id = self.pb_client.createVisualShape(p.GEOM_BOX,
                halfExtents=configuration["base_size"],
                rgbaColor=[1, 0, 0, 1],
            ),
            hip_collision_id = self.pb_client.createCollisionShape(p.GEOM_CYLINDER,
                radius=configuration["hip_size"][0],
                height=configuration["hip_size"][1],
            ),
            hip_visual_id = self.pb_client.createVisualShape(p.GEOM_CYLINDER,
                radius=configuration["hip_size"][0],
                length=configuration["hip_size"][1],
                rgbaColor=[1, 1, 0, 1],
            ),
            hip_thigh_collision_id = self.pb_client.createCollisionShape(p.GEOM_CYLINDER,
                radius=configuration["thigh_size"][0],
                height=2*configuration["thigh_size"][0],
            ),
            hip_thigh_visual_id = self.pb_client.createVisualShape(p.GEOM_CYLINDER,
                radius=configuration["thigh_size"][0],
                length=2*configuration["thigh_size"][0],
                rgbaColor=[1, 0, 0.6, 1],
            ),
            thigh_collision_id = self.pb_client.createCollisionShape(p.GEOM_CYLINDER,
                radius=configuration["thigh_size"][0],
                height=configuration["thigh_size"][1],
            ),
            thigh_visual_id = self.pb_client.createVisualShape(p.GEOM_CYLINDER,
                radius=configuration["thigh_size"][0],
                length=configuration["thigh_size"][1],
                rgbaColor=[0.5, 0.5, 0.5, 1],
            ),
            knee_collision_id = self.pb_client.createCollisionShape(p.GEOM_CYLINDER,
                radius=configuration["shin_size"][0],
                height=2*configuration["shin_size"][0],
            ),
            knee_visual_id = self.pb_client.createVisualShape(p.GEOM_CYLINDER,
                radius=configuration["shin_size"][0],
                length=2*configuration["shin_size"][0],
                rgbaColor=[1, 0, 0.6, 1],
            ),
            shin_collision_id = self.pb_client.createCollisionShape(p.GEOM_CYLINDER,
                radius=configuration["shin_size"][0],
                height=configuration["shin_size"][1],
            ),
            shin_visual_id = self.pb_client.createVisualShape(p.GEOM_CYLINDER,
                radius=configuration["shin_size"][0],
                length=configuration["shin_size"][1],
                rgbaColor=[0.5, 0.1, 0.5, 1],
            ),
            foot_collision_id = self.pb_client.createCollisionShape(p.GEOM_SPHERE,
                radius=configuration["foot_size"],
            ),
            foot_visual_id = self.pb_client.createVisualShape(p.GEOM_SPHERE,
                radius=configuration["foot_size"],
                rgbaColor=[0.1, 0.1, 0.1, 1],
            ),
        )
        return component_ids

    def set_default_joint_states(self):
        assert len(self.valid_joint_ids) == 12, "Expected 12 joints to control, got {}".format(len(self.valid_joint_ids))
        self.default_joint_states = np.zeros(12)
        for hip_idx in [0, 6]:
            self.default_joint_states[hip_idx] = np.pi/6
        for hip_idx in [3, 9]:
            self.default_joint_states[hip_idx] = -np.pi/6
        for thigh_idx in [1, 4, 7, 10]:
            self.default_joint_states[thigh_idx] = self.leg_bend_angle - np.pi/2
        for knee_idx in [2, 5, 8, 11]:
            self.default_joint_states[knee_idx] = np.pi - self.leg_bend_angle*2

    def reset_joint_states(self):
        for idx, j in enumerate(self.valid_joint_ids):
            self.pb_client.resetJointState(self.body_id, j, self.default_joint_states[idx])
    
    def get_inertial_data(self):
        """
        Return:
            dict:
                position: (3,)
                linear_velocity: (3,)
                rotation: (3,)
                angular_velocity: (3,)
        """
        position, orientation = self.pb_client.getBasePositionAndOrientation(self.body_id)
        linear_velocity, angular_velocity = self.pb_client.getBaseVelocity(self.body_id)
        return dict(
            position= np.array(position),
            linear_velocity= np.array(linear_velocity),
            rotation= np.array(p.getEulerFromQuaternion(orientation)),
            angular_velocity= np.array(angular_velocity),
        )



if __name__ == "__main__":
    # test script
    import time
    duration = 1/240
    pb_client = bullet_client.BulletClient(connection_mode=p.GUI)
    pb_client.setTimeStep(duration)
    pb_client.setGravity(0, 0, -9.8)
    pb_client.setAdditionalSearchPath(pb_data.getDataPath())
    pb_client.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
    robot = MetaQuadrupedRobot(
        base_mass = 14, #0., # fixed base.
        default_base_transform = np.array([0, 0, 0.3, 0, 0, 0, 1]),
        pb_client= pb_client,
    )
    cmd_limits = robot.get_cmd_limits()
    while True:
        cmd = np.random.uniform(low=cmd_limits[0], high=cmd_limits[1])
        robot.send_joints_cmd(cmd)
        pb_client.stepSimulation()
        time.sleep(duration)
