
import gym
import math
import numpy as np

from .asModel.asModel import ProModel,AsModel
from .asModel.pidCtrl import PidController
from .asModel.windModel import WindModel
from gym import error, spaces, utils

class asModelEnv(gym.Env):
    def __init__(self, **kwargs):
        super(asModelEnv, self).__init__()
        self.proModel = ProModel()
        self.asModel = AsModel()
        self.pidCtrler = PidController()
        self.wModel = WindModel()

        self.state_bounds = np.array([[-np.pi/4, -np.pi/4, -np.pi, -20, -20, -10, -5, -5, -5, 0, -8.7*1.5],\
                                        [np.pi/4, np.pi/4, np.pi, 20, 20, 10, 5, 5, 5, 2.5, 8.7*1.5]])

        self.X_bounds = np.array([100, 100, 10, 3/180.0*np.pi, 3/180.0*np.pi, np.pi, 10, 0, 0, 0, 0, 0,])
        self.U_bounds = np.array([[0,0,0,0,0,-8.7*0.2],[1,0,0,0,0,8.7*0.2]])
        #----action(psi-target)-----

        action_low = np.array([-1,-1])
        action_high = np.array([1,1])
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        #----state（x,y,h,phi,theta,psi,u,v,w,p,q,r）----
        self.observation_space = spaces.Box(-np.inf*np.ones(11,), np.inf*np.ones(11,), dtype=np.float32)

    def seed(self, seed):
        np.random.seed(seed)
        self.wModel.seed(seed)

    def stateInit(self):
        self.wModel.initWind()
        X = np.random.uniform(-1*self.X_bounds, self.X_bounds)
        U = np.random.uniform(self.U_bounds[0,:].ravel(), self.U_bounds[1,:].ravel())
        return X, U

    def modelStep(self, X, action):

        uRef = action[1]*150
        yawRef = action[0]*np.pi

        yaw = X[5]
        u = X[6]
        h = X[2]

        proller = self.pidCtrler.calThro(u, uRef, yaw, yawRef)
        U = self.proModel.calU(proller)
        W = self.wModel.calW(h)
        dx, alpha, beta = self.asModel.calDx(X, U, W)
        dt = 0.1
        #print('dx:',dx)
        k1 = dt*dx
        dx2,alpha, beta = self.asModel.calDx(X+k1*0.5, U, W)

        k2 = dt*dx2
        dx3, alpha, beta = self.asModel.calDx(X+k2*0.5, U, W)
        k3 = dt*dx3

        dx4, alpha, beta = self.asModel.calDx(X+k3, U, W)
        k4 = dt*dx4

        X = X+ (k1+2*k2+2*k3+k4)/6.0

        return X, U, alpha, beta

    def getWind(self):
        return self.wModel.getWind()
    def getPos(self):
        return self.asModel.getPos()
    def getYaw(self):
        return self.asModel.getYaw()
