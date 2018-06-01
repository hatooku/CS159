from training.postprocess import get_action
from training.pytorch_model import NeuralNet
import sys
sys.path.append('HFO')
import itertools
from hfo import *
import os
import torch
os.chdir('HFO')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "../training/models/15000_pytorch_1v0_partialstate_FINAL.model"

def main():
    # Create model
    model = NeuralNet()
    if device == 'cuda':
    	model = model.cuda()
    if device != 'cuda':
    	model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, location: storage))
    else:
    	model.load_state_dict(torch.load(MODEL_PATH))

    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(LOW_LEVEL_FEATURE_SET, 'bin/teams/base/config/formations-dt', 6000, 'localhost', 'base_left', False)



    for episode in itertools.count():
        status = IN_GAME
        while status == IN_GAME:
            # Grab the state features from the environment
            features = hfo.getState()
            # print(",".join(list(map(str, features))))
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)
            if device == 'cuda':
            	features_tensor = features_tensor.cuda()
            y = model(features_tensor)
            assert(y[0].shape[1] == 4)
            action_arrays = (y[0].data.cpu().numpy(), y[1].data.cpu().numpy())
            action = get_action(action_arrays)
            move = action[0]
            # print(action_arrays[0])
            # Take an action and get the current game status
            if move == 0:
                hfo.act(DASH, action[1], action[2])
                print("DASH", action)
            elif move == 1:
                hfo.act(TURN, action[1])
                print("TURN", action)
            elif move == 2:
                # hfo.act(TACKLE, action[1])
                # print("TACKLE", action)
                hfo.act(NOOP)
                print("TACKLE -> NOOP")
            else:
                hfo.act(KICK, action[1], action[2])
                print("KICK", action)
            # Advance the environment and get the game status
            status = hfo.step()
        # Check the outcome of the episode
        print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
        # Quit if the server goes down
        if status == SERVER_DOWN:
            hfo.act(QUIT)
            exit()

if __name__ == '__main__':
  main()
