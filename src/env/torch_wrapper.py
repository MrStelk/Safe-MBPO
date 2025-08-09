from gym import Wrapper
#import time
from ..torch_util import torchify, numpyify


class TorchWrapper(Wrapper):
    def reset(self):
        return torchify(self.env.reset())

    def step(self, action):
        #time.sleep(.002)
        observation, reward, done, info = self.env.step(numpyify(action))
        return torchify(observation), float(reward), done, info
