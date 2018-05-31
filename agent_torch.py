from training.postprocess import get_action
from training.pytorch_model import NeuralNet
import sys
sys.path.append('HFO')
import itertools
from hfo import *
import os
import torch
os.chdir('HFO')

def main():
    # Create model
    model = NeuralNet().cuda()
    model.load_state_dict(torch.load("../training/models/15000_pytorch_1v0_fullstate_FINAL.model"))

    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(LOW_LEVEL_FEATURE_SET, 'bin/teams/base/config/formations-dt', 6000, 'localhost', 'base_left', False)



    for episode in itertools.count():
        status = IN_GAME
        while status == IN_GAME:
            # Grab the state features from the environment
            features = hfo.getState()
            features_tensor = torch.from_numpy(features).cuda().float().unsqueeze(0)
            y = model(features_tensor)
            action_arrays = np.concatenate((y[0].data.cpu().numpy(), y[1].data.cpu().numpy()), axis=1)
            action = get_action(action_arrays)
            move = action[0]
            # Take an action and get the current game status
            if move == 0:
                hfo.act(DASH, action[1], action[2])
            elif move == 1:
                hfo.act(TURN, action[1])
            elif move == 2:
                hfo.act(TACKLE, action[1])
            else:
                hfo.act(KICK, action[1], action[2])
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
