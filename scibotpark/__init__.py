import gym

""" This is file helps register the environment in the gym registry.
Thus, you don't need to intall libraries that are not required in the environment.
"""
gym.register(
    "UnitreeLocomotion-v0",
    entry_point= "scibotpark.locomotion.unitree.locomotion:UnitreeForwardEnv",
)
gym.register(
    "Panda-v0",
    entry_point= "scibotpark.panda.base:PandaEnv",
)
gym.register(
    "PandaManipulation-v0",
    entry_point= "scibotpark.panda.manipulation.base:PandaManipulationEnv",
)