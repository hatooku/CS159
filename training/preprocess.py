import numpy as np
from key import OUTPUT_SIZE, DASH, DASH_POW, DASH_DEG, TURN, TURN_DEG, TACKLE, TACKLE_DEG, KICK, KICK_POW, KICK_DEG

LOG_FILE = 'base_left-11.log'
INPUT_SIZE = 68
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

# def fetch_data_old(log_file):
#     X = []
#     y = []
#     prev_iter = 0
#     features_exist = False
#     action_exists = False
#     prev_valid = False

#     with open(log_file) as f:
#         for line in f:
#             words = line.strip().split()
#             curr_iter = int(words[0])

#             if curr_iter != prev_iter:
#                 if prev_valid:
#                     X.append(state_features)
#                     y.append(action)
#                     prev_valid = False

#             if words[3] == 'StateFeatures':
#                 state_features = get_feature_array(words[4:])
#                 features_exist = True
#             else:
#                 action_exists = False
#                 for action in ACTIONS:
#                     if action in line:
#                         action_exists = True

#                 if action_exists and features_exist:
#                     action = get_action_array(words[4])
#                     prev_valid = True
#                 else:
#                     features_exist = False

#             prev_iter = curr_iter

#         if prev_valid:
#             X.append(state_features)
#             y.append(action)

#     return np.array(X), np.array(y)

def get_feature_array(features):
    feature_array = [float(feat) for feat in features]
    assert len(feature_array) == INPUT_SIZE

    for feat in feature_array:
        assert feat >= -1. and feat <= 1.

    return feature_array

def get_action_array(action):
    action_array = [0. for i in range(OUTPUT_SIZE)]
    param_str = action[action.index('(')+1:-1]
    action_type = action.split('(')[0]

    if action_type == 'Dash':
        params = [float(param) for param in param_str.split(',')]
        action_array[DASH] = 1.
        action_array[DASH_POW] = params[0]
        action_array[DASH_DEG] = params[1]
    elif action_type == 'Turn':
        param = float(param_str)
        action_array[TURN] = 1.
        action_array[TURN_DEG] = param
    elif action_type == 'Tackle':
        params = [float(param) for param in param_str.split(',')]
        action_array[TACKLE] = 1.
        action_array[TACKLE_DEG] = params[0]
    elif action_type == 'Kick':
        params = [float(param) for param in param_str.split(',')]
        action_array[KICK] = 1.
        action_array[KICK_POW] = params[0]
        action_array[KICK_DEG] = params[1]

    return action_array

if __name__ == '__main__':
    X, y = fetch_data(LOG_FILE)
