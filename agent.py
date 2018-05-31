import itertools
from HFO.hfo import *
from keras.models import load_model
from training.postprocess import get_action

def main():
    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(LOW_LEVEL_FEATURE_SET, 'bin/teams/base/config/formations-dt', 6000, 'localhost', 'base_left', False)
    for episode in itertools.count():
        status = IN_GAME
        while status == IN_GAME:
            # Grab the state features from the environment
            features = hfo.getState()
            model = load_model('training/models/basic_model.h5')
            action_array = model.predict(features)
            action = get_action(action_array)
            # Take an action and get the current game status
            if len(action) == 2:
                hfo.act(action[0], action[1])
            else:
                hfo.act(action[0], action[1], action[2])
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