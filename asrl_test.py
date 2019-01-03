import gym
import as_envs
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['ASRL_CONFIG_PATH'] = os.path.join(os.getcwd(),"as_envs/envs")

env = gym.make('airship_DirCtrl-v0')
alphas = np.array([])
betas = np.array([])
states, reward, done,k =  env.step(1)
states = states
alphas = np.append(alphas, k['alpha'])
betas = np.append(betas, k['beta'])
rewards = np.array([])
end_time = 0
np.append(rewards,reward)
max_step = 12000
for i in range(max_step):
    state, reward, done,k =  env.step(1)
    np.append(rewards,reward)
    #print(f"alpha:{k['alpha']}")
    #print(f"beta:{k['beta']}")
    states = np.vstack((states, state))
    alphas = np.append(alphas, k['alpha'])
    betas = np.append(alphas, k['beta'])
    if done:
        end_time = i
        break

f, ax = plt.subplots(8, sharex=True, sharey=False)
f.set_size_inches(10, 20)
ax[0].set_ylabel("phi")
ax[1].set_ylabel("theta")
ax[2].set_ylabel("psi")
ax[3].set_ylabel("u")
ax[4].set_ylabel("v")
ax[5].set_ylabel("w")
ax[6].set_ylabel("alpha")
ax[7].set_ylabel("beta")

ax[0].set_xlim([0, max_step])
res_linewidth = 2
linestyles = ["c", "m", "b", "g"]
reflinestyle = "k--"
error_linestyle = "r--"
for i in range(6):
    ax[i].plot(states[:,i], reflinestyle)

ax[6].plot(alphas, reflinestyle)

ax[7].plot(betas, reflinestyle)

plt.show()
