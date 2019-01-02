
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
        k1 = dt*dx
        dx2,alpha, beta = self.asModel.calDx(X+k1*0.5, U, W)

        k2 = dt*dx2
        dx3, alpha, beta = self.asModel.calDx(X+k2*0.5, U, W)
        k3 = dt*dx3

        dx4, alpha, beta = self.asModel.calDx(X+k3, U, W)
        k4 = dt*dx4

        X = X+ (k1+2*k2+2*k3+k4)/6.0
        
        return X, U, alpha, beta



    def getPos(self):
        return self.asModel.getPos()
    def getYaw(self):
        return self.asModel.getYaw()
