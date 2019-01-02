from gym.envs.registration import register

kwargs = {
    "max_sim_time": 20*60.,
    }

id = 'airship_DirCtrl-v0'
register(
    id=id,
    entry_point='as_envs.envs:asDirCtrlEnv',
    kwargs=kwargs)
