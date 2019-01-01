import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
window = 50

# data_dir = './Pong/dqn/losses'
# data_dir = './Pong/dqn/qvalues'
data_dir = './Pong/dqn/rewards'
# first = 1
# for i in range(5):
#     if i > 0:
#         first = 0

#     if data_dir == './Pong/dqn/losses':
#         fn = 'losses_dqn_seed' + str(i) + '.npy'
#         title = 'Loss per Episode (for 5 random seeds)'
#         label2 = 'Losses - DQN'
#         ylabel = 'Losses'

#     if data_dir == './Pong/dqn/qvalues':
#         fn = 'next_q_value_dqn_seed' + str(i) + '.npy'
#         title = 'Q-Values (for 5 random seeds)'
#         label2 = 'Predicted Q-Values - DQN'
#         ylabel = 'Q-Values'
        
#     if data_dir == './Pong/dqn/rewards':
#         fn = 'rewards_dqn_seed' + str(i) + '.npy'
#         title = 'Reward per Episode (for 5 random seeds)'
#         label2 = 'Rewards - DQN'
#         ylabel = 'Rewards'
        
#     path = os.path.join(data_dir, fn)
#     data = np.load(path)
#     data_list = np.ndarray.tolist(data)
#     print(len(data_list))
#     if first:
#         plt.plot(pd.Series(data_list).rolling(window).mean(), alpha = .5, c = 'r', label = label2)
#     else:
#         plt.plot(pd.Series(data_list).rolling(window).mean(), alpha = .5, c = 'r')


# data_dir = './Pong/ddqn/losses'
# data_dir = './Pong/ddqn/qvalues'
data_dir = './Pong/ddqn/rewards'

first = 1
for i in range(5):
    if i > 0:
        first = 0
    if data_dir == './Pong/ddqn/losses':
        fn = 'losses_ddqn_seed' + str(i) + '.npy'
        title = 'Loss per Episode (for 5 random seeds)'
        label = 'Losses - DDQN'
    
    if data_dir == './Pong/ddqn/qvalues':
        # fn = 'next_q_state_values_ddqn_seed' + str(i) + '.npy'
        fn = 'next_q_value_ddqn_seed' + str(i) + '.npy'
        title = 'Q-Values (for 5 random seeds)'
        label = 'Predicted Q-Values - DDQN'
    
    if data_dir == './Pong/ddqn/rewards':
        fn = 'rewards_ddqn_seed' + str(i) + '.npy'
        # fn = 'rewards.npy'
        title = 'Reward per Episode'
        label = 'Rewards - DDQN'

    path = os.path.join(data_dir, fn)
    data = np.load(path)
    data_list = np.ndarray.tolist(data)
    # plt.plot(pd.Series(reward_list).rolling(window).mean(), c = 'r')
    if first:
        plt.plot(pd.Series(data_list).rolling(window).mean(), alpha = .5, c = 'b', label = label)
    else:
        plt.plot(pd.Series(data_list).rolling(window).mean(), alpha = .5, c = 'b')


ylabel = 'Rewards'
plt.title(title)
plt.xlabel('Episodes')
plt.ylabel(ylabel)
plt.legend()
plt.show()
