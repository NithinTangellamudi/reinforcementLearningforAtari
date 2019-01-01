import math, random
import gym
from gym import wrappers
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from gym.wrappers import Monitor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
from IPython.display import clear_output
from training import DQN
from DemonAttack_BW import *
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
#################################
#TODO: current pipeline is to generate checkpoints on BW using training.py,
# then generate videos on a personal computer using generate_videos.py since BW doesn't have
# the right version of gym
# to get videos copy over the training checkpoints folder to the same directory as generate_videos.py,
# then run generate_videos.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env_id = "DemonAttackNoFrameskip-v0"
env=gym.make(env_id)
env.reset()
done=False
#env.render()
while not done:
    action = random.randrange(env.action_space.n)
    next_state,reward,done,_ = env.step(action)
    #env.render()
env.reset()
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
# env.render()
#################################
env_id = "DemonAttackNoFrameskip-v0"
# No exploring, only playin to the model for Testing
epsilon=0
# done=False
# set checkpoint load directory
# this is after training is done and all the checkpoints are in a folder + we want to generate videos
# easily generalize by setting env_id as a variable
ckpt_names_fn = 'checkpoint_paths.txt'
ckpt_dir = os.path.join('./checkpoints', env_id)
ckpt_names_path = os.path.join(ckpt_dir, ckpt_names_fn)
#with open(ckpt_names_path, 'r') as f:
 #   ckpt_paths = json.load(f)
vid_env_id_dir = os.path.join('./videos', env_id)
dir_exist = os.path.exists(vid_env_id_dir)
if not dir_exist:
    os.mkdir(vid_env_id_dir)
    
def test(model, video_path):
    done=False
    model.eval()
    env_id = "DemonAttackNoFrameskip-v0"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    env = gym.wrappers.Monitor(env, video_path, video_callable = False, force = True)
    
    #env = gym.make(env_id)
    #env = gym.wrappers.Monitor(env, video_path, video_callable = False, force = True)
    state = env.reset()
    #state = state.reshape((3,210,160))
    #state = np.expand_dims(state, 0)
    print(state.shape)
    
    #env.render()
    
    testReward=0
    while not done:
        action = model.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        #env.render()
        state = next_state
        #state = state.reshape((3,210,160))
        testReward += reward
    state = env.reset()
    if done:
        env.env.close()
#################################
#for ckpt in ckpt_paths:
    #ckpt_path = ckpt
    #ckpt_fn = ckpt_path.split('/')[-1]
    #print(ckpt_fn)
mod_path= "./bw_train/target_DemonAttackNoFrameskip-v0_10000000.model"
video_path = os.path.join('./videos/', env_id)
#dir_exist = os.path.exists(video_path)
#if (dir_exist == 0):
 #   os.mkdir(video_path)
# Load the Model
#current_model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
current_model = torch.load(mod_path, map_location='cpu')
# run the loaded model, save the output
test(current_model, video_path)
#################################
