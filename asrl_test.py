import gym
import as_envs
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['ASRL_CONFIG_PATH'] = os.path.join(os.getcwd(),"as_envs/envs")

env = gym.make('airship_DirCtrl-v0')
init_state = env.reset()

alphas = np.array([])
betas = np.array([])
lats = np.array([])
lons = np.array([])

states, reward, done,k =  env.step(np.array([90/360, 0]))
states = states
alphas = np.append(alphas, k['alpha'])
betas = np.append(betas, k['beta'])
lats = np.append(lats, k['x'])
lons = np.append(lons, k['y'])

rewards = np.array([])
end_time = 0
np.append(rewards,reward)
max_step = 20000
for i in range(max_step):
    state, reward, done,k =  env.step(np.array([90/360, 0]))
    np.append(rewards,reward)
    #print(f"alpha:{k['alpha']}")
    #print(f"beta:{k['beta']}")
    states = np.vstack((states, state))
    alphas = np.append(alphas, k['alpha'])
    betas = np.append(alphas, k['beta'])
    lats = np.append(lats, k['x'])
    lons = np.append(lons, k['y'])

    if done:
        end_time = i
        break

wVel = env.getWind()
print(f"wVel:{wVel}")

print("init_state",init_state)
print("final_state",states[-1,:])

f, ax = plt.subplots(3, sharex=True, sharey=False)
f.set_size_inches(10, 25)
ax[0].set_ylabel("phi")
ax[1].set_ylabel("theta")
ax[2].set_ylabel("psi")
#ax[3].set_ylabel("u")
#ax[4].set_ylabel("v")
#ax[5].set_ylabel("w")
#ax[6].set_ylabel("alpha")
#ax[7].set_ylabel("beta")
#ax[8].set_ylabel("lat")
#ax[9].set_ylabel("lon")

ax[2].set_xlim([0, max_step])
res_linewidth = 2
linestyles = ["c", "m", "b", "g"]
reflinestyle = "k--"
error_linestyle = "r--"
for i in range(3):
    ax[i].plot(states[:,i]*180.0/np.pi, reflinestyle)

#ax[6].plot(alphas, reflinestyle)

#ax[7].plot(betas, reflinestyle)

#ax[8].plot(lats, reflinestyle)

#ax[9].plot(lons, reflinestyle)

plt.show()
