import numpy as np

class WindModel(object):
    def __init__(self):
        #self.sigma_wVel = 2
        #self.sigma_wDir = 5/180.0*np.pi
        self.sigma_wVel= 0
        self.sigma_wDir =0
    def seed(self, seed):
        np.random.seed(seed)
    def initWind(self):
        self.wVel = np.random.uniform(0, 10,1)
        self.wDir = np.random.uniform(-np.pi, np.pi,1)
        #self.wVel = 3
        #self.wDir = 90/360*np.pi*2
    def getWind(self):
        return np.array([-1*self.wVel*np.cos(self.wDir),-1*self.wVel*np.sin(self.wDir),0])
    def calW(self,h):
        wVel = np.random.normal(self.wVel, self.sigma_wVel)
        wDir = np.random.normal(self.wDir, self.sigma_wDir)
        #return np.array([-1*wVel*np.cos(wDir),-1*wVel*np.sin(wDir),0])
        return np.array([0, -5, 0])
        #return np.array([0,0,0])
