import numpy as np

LOG_FILE = 'base_left-11.log'
OUTPUT_SIZE = 10
ACTIONS = {'Dash', 'Turn', 'Tackle', 'Kick'}

def fetch_data(log_file):
	X = []
	y = []
	i = 0

	with open(log_file) as f:
		lines = f.readlines()
	lines = [line.strip() for line in lines]

	while i < len(lines):
		words = lines[i].split()

		if words[3] == 'StateFeatures':
			action_exists = False
			for action in ACTIONS:
				if action in lines[i+1]:
					action_exists = True

			if action_exists:
				state_features = get_feature_array(words[4:])
				X.append(state_features)

				action = get_action_array(lines[i+1].split()[4])
				y.append(action)

				i += 2
			else:
				i += 1
		else:
			i += 1
	
	return np.array(X), np.array(y)

def get_feature_array(features):
	feature_array = [float(feat) for feat in features]
	for feat in feature_array:
		assert feat >= -1.0 and feat <= 1.0
	return feature_array

def get_action_array(action):
	action_array = [0 for i in range(OUTPUT_SIZE)]
	param_str = action[action.index('(')+1:-1]

	if action[:4] == 'Dash':
		params = [float(param) for param in param_str.split(',')]
		action_array[0] = 1
		action_array[1] = params[0]
		action_array[2] = params[1]
	elif action[:4] == 'Turn':
		param = float(param_str)
		action_array[3] = 1
		action_array[4] = param
	elif action[:6] == 'Tackle':
		params = [float(param) for param in param_str.split(',')]
		action_array[5] = 1
		action_array[6] = params[0]
	elif action[:4] == 'Kick':
		params = [float(param) for param in param_str.split(',')]
		action_array[7] = 1
		action_array[8] = params[0]
		action_array[9] = params[1]

	return action_array

if __name__ == '__main__':
	X, y = fetch_data(LOG_FILE)