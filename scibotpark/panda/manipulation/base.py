import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from scibotpark.panda.base import PandaEnv
from scibotpark.objects_models.pybullet import MultiObjectsMixin

class PandaManipulationEnv(MultiObjectsMixin, PandaEnv):
    pass


if __name__ == "__main__":
    from scibotpark.panda.utils import get_keyboard_action
    import time
    
    env = PandaManipulationEnv(
        object_names= [
            "YcbBanana",
            "YcbHammer",
        ],
        position_sample_region= None,
        robot_kwargs= dict(
            arm_control_mode= "translation",
        ),
        horizon= int(1e5),
        pb_client= bullet_client.BulletClient(connection_mode= p.GUI),
    )

    env.robot.set_poi_visiblity(True)
    env.reset()
    
    action = np.zeros_like(env.action_space.sample())
    while True:
        action = get_keyboard_action(env, action)
        obs, reward, done, info = env.step(action)
        if done:
            print("Environment done, reset...")
            obs = env.reset()

        time.sleep(1./200. * 1)

