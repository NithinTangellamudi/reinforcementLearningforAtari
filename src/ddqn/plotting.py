import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


data_dir = 'c:/home/classes/ie534-project/rewards_from_training/pong'
window = 100 # amount of episodes to average the reward over
for i in range(5):
    fn = 'next_q_value_ddqn_seed' + str(i) + '.npy'
    file_path = os.path.join(data_dir, fn)
    reward = np.load(file_path)
    reward_list = np.ndarray.tolist(reward)
    rewardPLOT = reward_list
    plt.plot(pd.Series(reward_list).rolling(window).mean(), c = 'r')
    # plt.savefig('next_q_value_dqn_seed')

    # fn = 'next_q_value_ddqn_seed' + str(i) + '.npy'
    # file_path = os.path.join(data_dir, fn)
    # reward = np.load(file_path)
    # reward_list = np.ndarray.tolist(reward)
    # rewardPLOT = reward_list
    # plt.plot(rewardPLOT, c = 'b')
    # plt.savefig('next_q_value_ddqn_seed')
    # plt.close()


plt.show()
