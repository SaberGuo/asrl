import gym
import as_envs
import os
import matplotlib.pyplot as plt
import numpy as np
import baselines.ddpg.ddpg as ddpg
import tensorflow as tf
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import make_vec_env

os.environ['ASRL_CONFIG_PATH'] = os.path.join(os.getcwd(),"as_envs/envs")

def main():
    seed = 5
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    #as_env = gym.make('airship_DirCtrl-v0')
    #print(f'shape:{as_env.observation_space.shape}')
    #as_env=gym.make('HalfCheetah-v2')
    env = make_vec_env('airship_DirCtrl-v0', 'mujoco', 1, seed, reward_scale=1)

    ddpg.learn(network='mlp',env = env,seed = seed, total_timesteps = 1e6)


if __name__ == "__main__":
    main()
