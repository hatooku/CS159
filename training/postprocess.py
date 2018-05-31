import numpy as np
try:
    from .key import *
except ImportError:
    from key import *

def get_action(action_arrays):
    actions = action_arrays[0][0]
    action_params = action_arrays[1][0]
    max_idx = np.argmax(actions)

    if max_idx == DASH:
        return (DASH, action_params[DASH_POW], action_params[DASH_DEG])
    elif max_idx == TURN:
        return (TURN, action_params[TURN_DEG])
    elif max_idx == TACKLE:
        return (TACKLE, action_params[TACKLE_DEG])
    else:
        return (KICK, action_params[KICK_POW], action_params[KICK_DEG])
