import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

from scibotpark.panda.base import PandaEnv
from scibotpark.objects_models.pybullet import MultiObjectsMixin

class PandaManipulationEnv(MultiObjectsMixin, PandaEnv):

    def __init__(self,
            *args,
            use_traybox= False,
            **kwargs,
        ):
        self.use_traybox = use_traybox
        super().__init__(*args, **kwargs)

    def _build_surroundings(self):
        return_ = super()._build_surroundings()
        if self.use_traybox:
            self.pb_client.loadURDF(
                osp.join(pb_data.getDataPath(), "tray/traybox.urdf"),
                [0.4, 0., 0.01,],
                [0., 0., 0., 1.],
                useFixedBase= True,
            )
        return return_


if __name__ == "__main__":
    from scibotpark.panda.utils import get_keyboard_action
    import time
    
    env = PandaManipulationEnv(
        use_traybox= True,
        object_names= [
            "YcbBanana",
            "YcbHammer",
        ],
        position_sample_region= None,
        robot_kwargs= dict(
            arm_control_mode= "trans_yaw",
            gripper_max_force= 20.,
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

