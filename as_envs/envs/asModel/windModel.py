import numpy as np

class WindModel(object):
    def __init__(self):
        pass

    def calW(self,h):
        wVel = 5
        wDir = 30.0/180.0*np.pi
        return np.array([np.sin(wDir)*wVel, np.cos(wDir)*wVel, 0])
        #return np.array([0,0,0])
