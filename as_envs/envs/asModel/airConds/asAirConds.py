import pandas as pd
from scipy import interpolate
import os
import numpy as np

class AsAirConds(object):
    def __init__(self, path="conds_cx.txt"):
        path = os.path.join(os.environ['ASRL_CONFIG_PATH'], "asModel/airConds/condsFile/", path)
        self.conds = pd.read_csv(path)
        #print(self.conds)
        a = self.conds['alpha'].values
        b = self.conds['beta'].values
        v = self.conds['value'].values
        self.func = interpolate.interp2d(a,b,v,kind="linear")


    def getValue(self, alpha, beta):
        '''
        alpha, beta: the unit is degree,range(0,12),range(0,45)
        '''
        return self.func(np.abs(alpha), np.abs(beta))
        #return [0]
if __name__ == "__main__":

    os.environ['ASRL_CONFIG_PATH'] = "/Users/guoxiao/Code/asrl/as_envs/envs/"
    conds = AsAirConds("conds_cx.txt")
    print(conds.getValue(0, 3))
