import logging
from .asModelEnv import asModelEnv
from gym import error, spaces, utils
import numpy as np
logger = logging.getLogger(__name__)


class asDirCtrlEnv(asModelEnv):
    def __init__(self, **kwargs):
        self.max_sim_time = kwargs["max_sim_time"]
        super(asDirCtrlEnv, self).__init__()
        self.target = np.array([0, 0])
        self.max_target = np.array([2000,2000])

        self.state_bounds = np.array([[-np.pi/4, -np.pi/4, -np.pi, -20, -20, -10, -5, -5, -5, 0, -8.7*1.5],\
                                        [np.pi/4, np.pi/4, np.pi, 20, 20, 10, 5, 5, 5, 2.5, 8.7*1.5]])
        #----action-----
        self.output_range = [-1,1]##yaw -PI~PI
        action_low = np.array([self.output_range[0]])
        action_high = np.array([self.output_range[1]])
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        #----state----
        self.X = np.zeros(12)
        self.X[4] = 0.0/180.0*np.pi
        self.X[6] = 0.0
        self.U = np.zeros(6)
        self.sim_step = 0
        state = self.state()

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=state.shape, dtype=np.float32)

    def compute_reward(self):
        """ Compute the reward """
        pos = self.getPos()
        pos = np.linalg.norm(pos - self.target)
        return pos


    def step(self, action):

        self.X, self.U, alpha, beta = self.modelStep(self.X, action)

        reward = self.compute_reward()
        pos = self.getPos()
        done =  np.abs(self.X[3])>15.0/180.0*np.pi or np.abs(self.X[4])>15/180.0*np.pi
        #np.any(pos>self.max_target) or self.sim_step >self.max_sim_time or
        self.sim_step = self.sim_step+1
        print(f"model_X:{self.X}")
        print(f"model_step:{self.sim_step}")
        print(f"done:{done}")
        return self.state(), reward, done, {"x": self.X[0], "y": self.X[1], "alpha": alpha, "beta": beta,"state_bounds":self.state_bounds}

    def getPos(self):
        return self.X[0:2]

    def getYaw(self):
        return self.X[5]

    def reset(self):
        self.X = np.zeros(12)
        self.U = np.zeros(6)
        self.sim_step = 0
        return self.state()

    def state(self):
        """ Get the current state """

        state = np.append(self.X[3:],self.U[0])
        state = np.append(state,self.U[5])
        #state = (state-self.state_bounds[0,:])/(self.state_bounds[1:]-self.state_bounds[0,:])*2-1
        return state
