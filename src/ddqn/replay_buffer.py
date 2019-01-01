from collections import deque
import numpy as np
import random

###############################
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



    
