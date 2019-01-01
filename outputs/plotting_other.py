import numpy as np
import matplotlib.pyplot as plt
import os 

# game = 'DemonAttack'
# ep = 57200000

game = 'Robotank'
ep = 48400000

# game = 'Zaxxon'
# ep = 29600000

plot = 'rewards'
# plot = 'losses'


data_dir = game
fn = plot + '_' + game + 'NoFrameskip-v0_' + str(ep) + '.npy'

path = os.path.join(data_dir, fn)
data = np.load(path)

data_list = np.ndarray.tolist(data)

plt.plot(data_list, alpha = .5, c = 'b')


title = game + ': ' + plot
plt.title(title)
plt.show()
