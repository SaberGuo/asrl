import logging
from .asModelEnv import asModelEnv

import numpy as np
logger = logging.getLogger(__name__)


class asDirCtrlEnv(asModelEnv):
    def __init__(self, **kwargs):
        self.max_sim_time = kwargs["max_sim_time"]
        super(asDirCtrlEnv, self).__init__()
        self.target = np.array([0, 0])
        self.max_target = np.array([2000,2000])
        self.X, self.U = self.stateInit()
        self.sim_step = 0


    def compute_reward(self):
        """ Compute the reward """
        pos = self.getPos()
        max_pos = np.linalg.norm(self.max_target-self.target)
        cur_pos = np.linalg.norm(pos - self.target)
        r = (max_pos-cur_pos)/max_pos
        return r


    def step(self, action):
        self.X, self.U, alpha, beta = self.modelStep(self.X, action)

        reward = self.compute_reward()
        pos = self.getPos()
        '''
        截止条件
        '''
        done =   self.sim_step >self.max_sim_time or np.abs(self.X[3])>15.0/180.0*np.pi or np.abs(self.X[4])>15/180.0*np.pi
        # np.any(pos>self.max_target) or

        self.sim_step = self.sim_step+1
        #print(f"model_X:{self.X}")
        #print(f"model_step:{self.sim_step}")
        #print(f"done:{done}")
        return self.state(), reward, done, {"x": self.X[0], "y": self.X[1], "alpha": alpha, "beta": beta,"state_bounds":self.state_bounds}

    def getPos(self):
        return self.X[0:2]

    def getYaw(self):
        return self.X[5]

    def reset(self):
        self.X, self.U = self.stateInit()
        print("self.U:", self.U)
        self.sim_step = 0

        return self.state()

    def state(self):
        """ Get the current state """

        state = np.append(self.X[3:],self.U[0])
        state = np.append(state,self.U[5])
        state = (state-self.state_bounds[0,:].ravel())/(self.state_bounds[1:].ravel()-self.state_bounds[0,:].ravel())*2-1
        return state
