import math, random

import gym
from gym import wrappers
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from gym.wrappers import Monitor

from wrappers import make_atari, wrap_deepmind, wrap_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
from IPython.display import clear_output
from dqn_class import CnnDQN

#################################
# No exploring, only playin to the model for Testing
epsilon=0

def test(model):
    done=0
    state = env.reset()

    testReward=0
    first = 1
    while not done:
        env.render()
        if first:
            input('Press any Key to Continue!')
            first = 0
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        #env.render()
        state = next_state
        testReward += reward
    state = env.reset()
    if done:
        env.close()

#################################
base_dir = 'C:/home/classes/ie534-project/outputs'

# choose your environment
# env_id = 'PongNoFrameskip-v0'
# env_id = "DemonAttackNoFrameskip-v0"
# env_id = 'RobotankNoFrameskip-v0'
# env_id = 'ZaxxonNoFrameskip-v0'
# env_id = 'IceHockeyNoFrameskip-v0'
# env_id = 'MsPacmanNoFrameskip-v0'
# env_id = 'MontezumaRevengeNoFrameskip-v0'

env=gym.make(env_id)

# choose your model
# path = 'Pong/ddqn/models/current21600000_seed0_DDQN.model'
# path = 'DemonAttack/target_DemonAttackNoFrameskip-v0_57200000.model'
# path = 'Robotank/target_RobotankNoFrameskip-v0_48400000.model'
# path = 'Zaxxon/target_ZaxxonNoFrameskip-v0_29600000.model'

# path = './models/target_BreakoutNoFrameskip-v0_9700000.model'
# path = './models/target_IceHockeyNoFrameskip-v0_9700000.model'
# path = './models/target_MontezumaRevengeNoFrameskip-v0_9700000.model'
# path = './models/target_MsPacmanNoFrameskip-v0_19800000.model'
# path = './models/target_PongNoFrameskip-v0_9900000.model'


ckpt_path = os.path.join(base_dir, path)
# ckpt_path = 'C:/home/classes/ie534-project/src/ddqn/models/target_DemonAttackNoFrameskip-v0_47700000.model'


# Load the Model
current_model = torch.load(ckpt_path, map_location='cpu')
current_model.eval()

env = wrap_deepmind(env)
env = wrap_pytorch(env)


# run the loaded model, save the output
test(current_model)

