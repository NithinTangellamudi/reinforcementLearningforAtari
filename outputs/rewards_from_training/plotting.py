import numpy as np
import matplotlib.pyplot as plt
import os 


data_dir_1 = 'c:/home/classes/ie534-project/outputs/rewards'
data_dir_2 = 'c:/home/classes/ie534-project/rewards_from_training'


fn = 'rewards_MontezumaRevengeNoFrameskip-v0.npy'


file_path_1 = os.path.join(data_dir_1, fn)
file_path_2 = os.path.join(data_dir_2, fn)


reward_1 = np.load(file_path_1)
reward_2 = np.load(file_path_2)

reward_list_1 = np.ndarray.tolist(reward_1)
reward_list_2 = np.ndarray.tolist(reward_2)

rewardPLOT_1 = reward_list_1
rewardPLOT_2 = reward_list_2

plt.plot(rewardPLOT_1)
plt.show()

plt.plot(rewardPLOT_2)
plt.show()