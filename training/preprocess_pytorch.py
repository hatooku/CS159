import numpy as np
try:
    from .key import *
except ImportError:
    from key import *

LOG_FILE = 'base_left-11.log'
ACTIONS = {'Dash', 'Turn', 'Tackle', 'Kick'}

def fetch_data(logs):
    """Input: Array of log filenames"""
    assert(isinstance(logs, list))
    X = None
    y = None
    for log_file in logs:
        if X is None:
            X, y = fetch_data_helper(log_file)
        else:
            X_new, y_new = fetch_data_helper(log_file)
            X = np.concatenate((X, X_new))
            y = np.concatenate((y, y_new))

    return X, y

def fetch_data_helper(log_file):
    X = []
    y = []
    prev_time = 0
    curr_features = None # string
    curr_actions = None # string

    with open(log_file) as f:
        for line in f:
            words = line.strip().split()
            curr_time = int(words[0])

            if curr_time != prev_time:
                if curr_features is not None and curr_actions is not None:
                    X.append(get_feature_array(curr_features))
                    y.append(get_action_array(curr_actions))
                prev_time = curr_time
                curr_features = None
                curr_actions = None

            if words[3] == "StateFeatures": # States
                curr_features = words[4:]
            elif words[1] == "8": # Actions
                curr_actions = words[4]

    if curr_features is not None and curr_actions is not None:
        X.append(get_feature_array(curr_features))
        y.append(get_action_array(curr_actions))

    return np.array(X), np.array(y)

def get_feature_array(features):
    feature_array = [float(feat) for feat in features]

    for feat in feature_array:
        assert feat >= -1. and feat <= 1.

    return feature_array

def get_action_array(action):
    action_array = [0. for i in range(OUTPUT_SIZE)]
    param_str = action[action.index('(')+1:-1]
    action_type = action.split('(')[0]

    if action_type == 'Dash':
        params = [float(param) for param in param_str.split(',')]
        action_array[0] = DASH
        action_array[DASH_POW] = params[0]
        action_array[DASH_DEG] = params[1]
    elif action_type == 'Turn':
        param = float(param_str)
        action_array[0] = TURN
        action_array[TURN_DEG] = param
    elif action_type == 'Tackle':
        params = [float(param) for param in param_str.split(',')]
        action_array[0] = TACKLE
        action_array[TACKLE_DEG] = params[0]
    elif action_type == 'Kick':
        params = [float(param) for param in param_str.split(',')]
        action_array[0] = KICK
        action_array[KICK_POW] = params[0]
        action_array[KICK_DEG] = params[1]

    return action_array

if __name__ == '__main__':
    X, y = fetch_data(LOG_FILE)
