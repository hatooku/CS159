import numpy as np
try:
    from .key import *
except ImportError:
    from key import *

def get_action(action_arrays):
    actions = action_arrays[0][0]
    action_params = action_arrays[1][0]

    max_idx = np.argmax(actions)
    # if max_idx == TACKLE:
    #     max_idx = TURN


    if max_idx == DASH:
        return (DASH, action_params[DASH_POW-1], action_params[DASH_DEG-1])
    elif max_idx == TURN:
        return (TURN, action_params[TURN_DEG-1])
    elif max_idx == TACKLE:
        return (TACKLE, action_params[TACKLE_DEG-1])
    else:
        return (KICK, action_params[KICK_POW-1], action_params[KICK_DEG-1])
