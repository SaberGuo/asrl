
import gym
import math
import numpy as np

from .asModel.asModel import ProModel,AsModel
from .asModel.pidCtrl import PidController
from .asModel.windModel import WindModel


class asModelEnv(gym.Env):
    def __init__(self, **kwargs):
        super(asModelEnv, self).__init__()
        self.proModel = ProModel()
        self.asModel = AsModel()
        self.pidCtrler = PidController()
        self.wModel = WindModel()



    def modelStep(self, X, yawRef):
        yaw = X[5]
        h = X[2]
        proller = self.pidCtrler.calThro(yaw, yawRef)
        U = self.proModel.calU(proller)
        W = self.wModel.calW(h)
        dx, alpha, beta = self.asModel.calDx(X, U, W)
        dt = 0.1
        #print(f'dx:{dx}')
        X = X+ dt*dx
        return X, U, alpha, beta



    def getPos(self):
        return self.asModel.getPos()
    def getYaw(self):
        return self.asModel.getYaw()
