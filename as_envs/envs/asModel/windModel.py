import numpy as np

class WindModel(object):
    def __init__(self):
        self.sigma_wVel = 2
        self.sigma_wDir = 5/180.0*np.pi
    def seed(self, seed):
        np.random.seed(seed)
    def initWind(self):
        self.wVel = np.random.uniform(0, 10,1)
        self.wDir = np.random.uniform(-np.pi, np.pi,1)
    def getWind(self):
        return self.wVel, self.wDir
    def calW(self,h):
        wVel = np.random.normal(self.wVel, self.sigma_wVel)
        wDir = np.random.normal(self.wDir, self.sigma_wDir)
        return np.array([-np.sin(wDir)*wVel, -np.cos(wDir)*wVel, 0])
        #return np.array([0,0,0])
