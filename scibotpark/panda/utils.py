import os
import os.path as osp
import numpy as np

import pybullet

def get_keyboard_action(env, action):
    """ This is a helper function used for env testing. Please don't use it outside this file.
    """
    keys = env.pb_client.getKeyboardEvents()
    z_p_key = ord('r')
    z_m_key = ord('f')
    x_p_key = ord('k')
    x_m_key = ord('i')
    y_p_key = ord('l')
    y_m_key = ord('j')
    open_key = ord('o')
    close_key = ord('p')
    number_key = [ord(k) for k in ['0','1','2','3','4','5','6','7','8','9']]
    action[:3] = np.zeros(3)
    if x_p_key in keys and keys[x_p_key] & pybullet.KEY_WAS_TRIGGERED:
        action[0] = 0.3
    if x_m_key in keys and keys[x_m_key] & pybullet.KEY_WAS_TRIGGERED:
        action[0] = -0.3
    if y_p_key in keys and keys[y_p_key] & pybullet.KEY_WAS_TRIGGERED:
        action[1] = 0.3
    if y_m_key in keys and keys[y_m_key] & pybullet.KEY_WAS_TRIGGERED:
        action[1] = -0.3
    if z_p_key in keys and keys[z_p_key] & pybullet.KEY_WAS_TRIGGERED:
        action[2] = 0.3
    if z_m_key in keys and keys[z_m_key] & pybullet.KEY_WAS_TRIGGERED:
        action[2] = -0.3
    if open_key in keys and keys[open_key] & pybullet.KEY_WAS_TRIGGERED:
        action[-1] = 1
    if close_key in keys and keys[close_key] & pybullet.KEY_WAS_TRIGGERED:
        action[-1] = -1
    for i, number in enumerate(number_key):
        if number in keys and keys[number] & pybullet.KEY_WAS_TRIGGERED:
            action[-1] = -1 + 0.2 * i
    return action