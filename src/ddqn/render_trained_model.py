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

def test(model, video_path):
    done=0
    state = env.reset()

    testReward=0
    first = 1
    while not done:
        env.render()
        if first:
            input()
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
env_id = "DemonAttackNoFrameskip-v0"
env=gym.make(env_id)

ckpt_fn = 'target_DemonAttackNoFrameskip-v0_47700000.model'
ckpt_path = 'C:/home/classes/ie534-project/src/ddqn/models/target_DemonAttackNoFrameskip-v0_47700000.model'


video_dir = 'my_video'
video_path = os.path.join('./videos', env_id, video_dir)

dir_exist = os.path.exists(video_path)
if (dir_exist == 0):
    os.mkdir(video_path)

# Load the Model
current_model = torch.load(ckpt_path, map_location='cpu')
current_model.eval()

env = wrap_deepmind(env)
env = wrap_pytorch(env)
env = gym.wrappers.Monitor(env, video_path, video_callable = False, force = True)


# run the loaded model, save the output
test(current_model, video_path)

