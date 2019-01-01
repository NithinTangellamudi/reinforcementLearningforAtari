st = 1
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
#         plt.plot(pd.Series(