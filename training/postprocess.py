import numpy as np
from .key import *

def get_action(action_array):
    actions = np.array([action_array[DASH], action_array[TURN], action_array[TACKLE], action_array[KICK]])
    max_idx = np.argmax(actions)

    if max_idx == 0:
        return (DASH, action_array[DASH_POW], action_array[DASH_DEG])
    elif max_idx == 1:
        return (TURN, action_array[TURN_DEG])
    elif max_idx == 2:
        return (TACKLE, action_array[TACKLE_DEG])
    else:
        return (KICK, action_array[KICK_POW], action_array[KICK_DEG])
