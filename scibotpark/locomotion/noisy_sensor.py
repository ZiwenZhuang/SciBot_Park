import numpy as np
import pybullet as p
import pybullet_data as pb_data
from pybullet_utils import bullet_client

class NoisySensorMixin:
    """ Mixin implementation for adding noise to env observation
    It does not change the action space nor observation space.
    NOTE: The subclass to use this mixin implementation are required to meet LocomotionEnv interface
    """
    def __init__(self,
            *args,
            noise_stds,
            **kwargs,
        ):
        """
        Current implementation only support Gaussian noise. TODO: add other noise options
        Args:
            noise_stds: Assuming you know the shape of your observation, this parameter must provide
                an array. If the observation is a dict, the stds must also be a dict.
                If you don't want noise, you can set a specific term to 0.
        """
        super().__init__(*args, **kwargs)
        self.noise_stds = noise_stds if isinstance(noise_stds, np.ndarray) else np.array(noise_stds)

    @staticmethod
    def add_noise(obs, std):
        """ Designed to be a recursive method to add noise to the original observation """
        if isinstance(obs, dict):
            assert isinstance(std, dict), "You must provide std as a dictionary, because the obs is a dict"
            return_ = dict()
            for k in obs.keys():
                return_[k] = NoisySensorMixin.add_noise(obs[k], std[k])
            return return_
        else:
            return np.random.normal(obs, std, size= obs.shape).astype(np.float32)

    def _get_obs(self):
        base_obs = super()._get_obs()
        noisy_obs = self.add_noise(base_obs, self.noise_stds)
        return noisy_obs
