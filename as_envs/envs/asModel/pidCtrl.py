from .config.Config import Config
import math
import numpy as np

class PidController(object):
    def __init__(self, path="pidConf.json"):
        self.conf = Config(path)
        self.CtrlSys_error_yaw_last = 0
        self.CtrlSys_error_v_last = 0
        self.CtrlSys_Thro_last = np.array([0,0,0,0])
        self.CtrlSys_ei2_y = 0
        self.CtrlSys_ei2_v = 0
        self.CtrlSys_Pro_Max = self.conf.CtrlSys_Pro_Max
        self.CtrlSys_Pro_Min = self.conf.CtrlSys_Pro_Min

    def _calYawError(self, yaw, yawRef):
        yaw = 1800*yaw/math.pi
        yawRef = 1800*yawRef/math.pi

        return yawRef - yaw

    def _ActionLimit(self, action, max, min):
        action[action>max] = max
        action[action<min] = min
        return action

    def _ActionChangeLimit(self, action, action_old, change_limit):
        signf = lambda x: 1 if x else -1
        tmp = (action>action_old)
        sig = np.abs(action-action_old)>change_limit
        tmp = change_limit*np.array([signf(xi) for xi in tmp])
        action[sig] = action_old[sig]+ tmp[sig]
        return action
    def _calActionForVel(self, u, uRef):
        '''
        u, uRef: m/s
        '''
        kvp1 = self.conf.kvp1
        kvp2 = self.conf.kvp2
        kvd = self.conf.kvd
        kvi = self.conf.kvi
        kvpt = self.conf.kvpt
        kvth = self.conf.kvth
        kvlimit = self.conf.kvlimit
        CtrlSys_pvth = self.conf.CtrlSys_pvth
        CtrlSys_ivth = self.conf.CtrlSys_ivth
        CtrlSys_error_v_last = self.CtrlSys_error_v_last
        CtrlSys_temp_sv = 0
        CtrlSys_ei2_v = self.CtrlSys_ei2_v
        CtrlSys_cv = u
        CtrlSys_tav = uRef
        #-------------------------速度误差------------------------------%
        CtrlSys_error_v = CtrlSys_tav - CtrlSys_cv;

        #-------------------如果没有超过门限值，则基础油门变化为0-----------------%
        if CtrlSys_error_v < kvth * 10.0 and CtrlSys_error_v > -kvth*10.0:
            CtrlSys_temp_sv = 0;
            CtrlSys_ei2_v= 0;
            #输出
            CtrlSys_sv = CtrlSys_temp_sv;

        #-------------------如果超过门限值，则采用PID算法计算输出值-----------------%
        else:
        #----------------------计算比例的值------------------------
            if CtrlSys_error_v > CtrlSys_pvth or CtrlSys_error_v < -CtrlSys_pvth:
            #如果在第二段内
                if CtrlSys_error_v > 0:
                    CtrlSys_temp_sv = kvp1 * CtrlSys_pvth + kvp2 * ( CtrlSys_error_v - CtrlSys_pvth )
                else:
                    CtrlSys_temp_sv = -kvp1 * CtrlSys_pvth + kvp2 * ( CtrlSys_error_v + CtrlSys_pvth )
            else:
            #--------------------如果在第一段内---------------------------
                CtrlSys_temp_sv = kvp1 * CtrlSys_error_v

        #-------------------计算积分的值----------------------------
        if CtrlSys_error_v < CtrlSys_ivth and CtrlSys_error_v > -CtrlSys_ivth:
        #-----------计算误差的积分----------------
            CtrlSys_ei2_v = CtrlSys_ei2_v + CtrlSys_error_v/10.0
            if CtrlSys_ei2_v>1000:
                CtrlSys_ei2_v =1000
            elif CtrlSys_ei2_v<-1000:
                CtrlSys_ei2_v = -1000
            CtrlSys_temp_sv = CtrlSys_temp_sv + CtrlSys_ei2_v * kvi / 1000.0
        else:
        #------------超出积分开启的范围，积分无效，并设置误差的积分为0--------
            CtrlSys_ei2_v = 0;
        self.CtrlSys_ei2_v = CtrlSys_ei2_v
        #----------------------计算微分的值---------------------
        CtrlSys_cdv =  CtrlSys_error_v_last - CtrlSys_error_v
        CtrlSys_temp_sv = CtrlSys_temp_sv - CtrlSys_cdv * kvd

        #---------------总比例控制-----------------
        CtrlSys_temp_sv = CtrlSys_temp_sv * kvpt

        #--------------输出限位---------------
        if CtrlSys_temp_sv > kvlimit:
            CtrlSys_temp_sv = kvlimit

        #--------------------输出----------------------
        CtrlSys_sv = CtrlSys_temp_sv;

        CtrlSys_error_v_last = CtrlSys_error_v


        return CtrlSys_sv

    def _calActionForYaw(self, yaw, yawRef):
        '''
        yaw,yawRef: rad
        '''
        CtrlSys_ei2_y = self.CtrlSys_ei2_y
        CtrlSys_error_yaw_last = self.CtrlSys_error_yaw_last
        CtrlSys_error_yaw = self._calYawError(yaw, yawRef)
        CtrlSys_pyth = self.conf.CtrlSys_pyth
        CtrlSys_iyth = self.conf.CtrlSys_iyth

        kyp1 = self.conf.kyp1
        kyp2 = self.conf.kyp2
        kyd = self.conf.kyd
        kyi = self.conf.kyi
        kypt = self.conf.kypt
        kyth = self.conf.kyth
        kylimit = self.conf.kylimit
        ##如果没有超过门限值，则方向舵角为0
        if np.abs(CtrlSys_error_yaw) < kyth *10:
            CtrlSys_temp_sy = 0
            CtrlSys_ei2_y = 0

            CtrlSys_sy = CtrlSys_temp_sy
        ##如果超过门限值，则采用PID算法计算输出值
        else:
            if CtrlSys_error_yaw > 1800:
                CtrlSys_error_yaw = -(3600 - CtrlSys_error_yaw)
            elif CtrlSys_error_yaw < -1800:
                CtrlSys_error_yaw = 3600 + CtrlSys_error_yaw
            ##--------------------计算比例的值P--------------------
            if (CtrlSys_error_yaw > CtrlSys_pyth) or (CtrlSys_error_yaw < -CtrlSys_pyth):
            #--------------------如果在第二段内--------------------
                if CtrlSys_error_yaw > 0:
                    CtrlSys_temp_sy = kyp1 * CtrlSys_pyth + kyp2 * ( CtrlSys_error_yaw - CtrlSys_pyth )
                else:
                    CtrlSys_temp_sy = -kyp1 * CtrlSys_pyth + kyp2 * ( CtrlSys_error_yaw + CtrlSys_pyth )

            else:
            #--------------------如果在第一段内---------------------------
                CtrlSys_temp_sy = kyp1 * CtrlSys_error_yaw

            ##--------------------计算比例的值I--------------------
            if (CtrlSys_error_yaw < CtrlSys_iyth) and (CtrlSys_error_yaw > -CtrlSys_iyth):
                #-----------计算误差的积分----------------
                CtrlSys_ei2_y = CtrlSys_ei2_y + CtrlSys_error_yaw/10.0
                if CtrlSys_ei2_y>1000:
                    CtrlSys_ei2_y =1000
                elif CtrlSys_ei2_y<-1000:
                    CtrlSys_ei2_y = -1000
                CtrlSys_temp_sy = CtrlSys_temp_sy + CtrlSys_ei2_y * kyi / 1000.0
            else:
                ##------------超出积分开启的范围，积分无效，并设置误差的积分为0--------
                CtrlSys_ei2_y = 0
            self.CtrlSys_ei2_y = CtrlSys_ei2_y
            ##----------------------计算微分的值---------------------
            CtrlSys_cdy =  CtrlSys_error_yaw_last - CtrlSys_error_yaw
            CtrlSys_temp_sy = CtrlSys_temp_sy - CtrlSys_cdy * kyd

            ##---------------总比例控制-----------------
            CtrlSys_temp_sy = CtrlSys_temp_sy * kypt
            #--------------输出限位---------------
            if CtrlSys_temp_sy > kylimit:
               CtrlSys_temp_sy = kylimit
            elif CtrlSys_temp_sy < -kylimit:
               CtrlSys_temp_sy = -kylimit
            CtrlSys_sy = CtrlSys_temp_sy
        self.CtrlSys_error_yaw_last = CtrlSys_error_yaw
        return CtrlSys_sy
    def calThro(self, u, uRef, yaw, yawRef):

        #AirscrewNCtrl = np.array(self.conf.AirscrewNCtrl) + self._calActionForVel(u, uRef)
        #print(f"AirscrewNCtrl:{AirscrewNCtrl}")
        AirscrewNCtrl = np.array(self.conf.AirscrewNCtrl)*(1+uRef)
        CtrlSys_sy = self._calActionForYaw(yaw, yawRef)
        ProCtrlPara  = self.conf.CtrlAllocatePara[0]
        CtrlSys_Thro = AirscrewNCtrl+np.array([1,1,-1,-1])*ProCtrlPara*CtrlSys_sy

        CtrlSys_Thro = self._ActionChangeLimit(CtrlSys_Thro, self.CtrlSys_Thro_last, self.conf.Thro_Change_Limit)
        CtrlSys_Thro = self._ActionLimit(CtrlSys_Thro, self.CtrlSys_Pro_Max, self.CtrlSys_Pro_Min)
        self.CtrlSys_Thro_last = CtrlSys_Thro
        return CtrlSys_Thro
