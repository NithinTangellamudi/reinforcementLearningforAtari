import math, random

import gym
import numpy as np
from numpy import shape
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

###############################
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

###############################
# replay buffer

# class ReplayBuffer(object):
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         state      = np.expand_dims(state, 0)
#         next_state = np.expand_dims(next_state, 0)
            
#         self.buffer.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
#     def __len__(self):
#         return len(self.buffer)

class ReplayBuffer(object):
    # uses a better version of Replay Buffer as shown in
    # Zhang - A Deeper Look at Experience Replay (2018)
    # In addition to randomly sampling transition tuples (s, a, r, s'),
    # we always return the most recent transition as part of the sampled 
    # transitions
    def __init__(self, capacity):
        '''
        Replay Buffer
        Params
        ------
        capacity (int): max number of transitions stored in the buffer.  
        When the number of transitions stored in the buffer is equal to capacity,
        the data is taken out (in the training loop), 
        '''
        # Deque is like a list, but it only accepts new items at either end.
        # By specifying maxlen, new items added to one end of a deque will
        # "push" items at the other end out of the data structure.  The
        # most recent transition will always be at the end of the buffer.
        self.buffer = deque(maxlen = capacity)
    
    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, state_next, done):
        '''Add a transition tuple to the end of the buffer'''
        state = np.expand_dims(state, 0)
        state_next = np.expand_dims(state_next, 0)

        self.buffer.append([state, action, reward, state_next, done])
    
    def sample(self, batch_size):
        # we have batch_size - 1 in order to make room for the most recent state
        state, action, reward, state_next, done = zip(*random.sample(self.buffer, batch_size-1))
        
        # put transition in useful data structures
        state, state_next = np.array(state).squeeze(), np.array(state_next).squeeze()
        action, reward, done = list(action), list(reward), list(done)

        state_curr, action_curr, reward_curr, state_next_curr, done_curr = self.buffer[-1]

        # convert to np arrays
        state_curr, state_next_curr = np.array(state_curr), np.array(state_next_curr)

        state = np.concatenate((state, state_curr))
        action.append(action_curr)
        reward.append(reward_curr)
        state_next = np.concatenate((state_next, state_next_curr))
        done.append(done_curr)

        return state, action, reward, state_next, done 
        
    
###############################
#env_id = "CartPole-v0"
env_id = "Pong-v0"
env = gym.make(env_id)

###############################
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
plt.plot([epsilon_by_frame(i) for i in range(10000)])
###############################
class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            action = random.randrange(env.action_space.n)
            '''
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            print(shape(state))
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
            '''
        else:
            action = random.randrange(env.action_space.n)
        return action
###############################

model = DQN(env.observation_space.shape[0], env.action_space.n)

if USE_CUDA:
    model = model.cuda()
    
optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayBuffer(1000)
###############################

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss
###############################
def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
###############################
num_frames = 10000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if frame_idx % 500 == 0:
        plot(frame_idx, all_rewards, losses)




















