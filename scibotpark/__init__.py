import gym

""" This is file helps register the environment in the gym registry.
Thus, you don't need to intall libraries that are not required in the environment.
"""
gym.register(
    "UnitreeLocomotion-v0",
    entry_point= "scibotpark.quadruped_robot.unitree.locomotion.base:UnitreeLocomotionEnv",
)
gym.register(
    "Panda-v0",
    entry_point= "scibotpark.panda.base:PandaEnv",
)
gym.register(
    "PandaManipulation-v0",
    entry_point= "scibotpark.panda.manipulation.base:PandaManipulationEnv",
)