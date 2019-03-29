from .config.Config import Config
import math
import numpy as np

class PidController(object):
    def __init__(self, path="pidConf.json"):
        self.conf = Config(path)
        self.CtrlSys_error_yaw_last = 0
        self.CtrlSys_error_v_last = 0
        self.CtrlSys_error_pit_last = 0
        self.CtrlSys_error_roll_last = 0
        self.CtrlSys_Thro_last = np.array([0,0,0,0])
        self.CtrlSys_ei2_y = 0
        self.CtrlSys_ei2_v = 0
        self.CtrlSys_ei2_p = 0
        self.CtrlSys_ei2_r = 0
        self.CtrlSys_Pro_Max = self.conf.CtrlSys_Pro_Max
        self.CtrlSys_Pro_Min = self.conf.CtrlSys_Pro_Min
        self.Proller_front_AM= np.array([[0,-1,1],[0,1,1],[0,-1,-1],[0,1,-1]])

    def _calAngleError(self, angleCur, angleRef):
        angleCur = 1800*angleCur/math.pi
        angleRef = 1800*angleRef/math.pi

        return angleRef - angleCur

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
        CtrlSys_tay = yawRef * 1800/np.pi
        CtrlSys_cy = yaw * 1800/np.pi
        CtrlSys_ei2_y = self.CtrlSys_ei2_y
        CtrlSys_error_yaw_last = self.CtrlSys_error_yaw_last
        CtrlSys_pyth = self.conf.CtrlSys_pyth
        CtrlSys_iyth = self.conf.CtrlSys_iyth

        kyp1 = self.conf.kyp1
        kyp2 = self.conf.kyp2
        kyd = self.conf.kyd
        kyi = self.conf.kyi
        kypt = self.conf.kypt
        kyth = self.conf.kyth
        kylimit = self.conf.kylimit
        #-------------------------偏航角误差------------------------------%
        if CtrlSys_tay<0:
            CtrlSys_tay = CtrlSys_tay + 3600
        if CtrlSys_tay>3600:
            CtrlSys_tay = CtrlSys_tay-3600

        if CtrlSys_cy<0:
            CtrlSys_cy = CtrlSys_cy + 3600
        if CtrlSys_cy>3600:
            CtrlSys_cy = CtrlSys_cy-3600

        CtrlSys_error_yaw = CtrlSys_tay - CtrlSys_cy

        #-------------------如果没有超过门限值，则方向舵角为0-----------------%
        if CtrlSys_error_yaw < kyth * 10.0 and CtrlSys_error_yaw > -kyth*10.0:
            CtrlSys_temp_sy = 0
            CtrlSys_ei2_y= 0

            #输出
            CtrlSys_sy = CtrlSys_temp_sy

        #-------------------如果超过门限值，则采用PID算法计算输出值-----------------%
        else:
            if CtrlSys_error_yaw > 1800:
               #ay 在cy 的左侧e为负,例ay=3300,CtrlSys_cy=300
                CtrlSys_error_yaw = -(3600 - CtrlSys_error_yaw)
            if CtrlSys_error_yaw < -1800:
               #ay在cy的右侧e为正,例ay =300,CtrlSys_cy=3300
                CtrlSys_error_yaw = 3600 + CtrlSys_error_yaw

            #----------------------计算比例的值------------------------
            if CtrlSys_error_yaw > CtrlSys_pyth or CtrlSys_error_yaw < -CtrlSys_pyth:
                #如果在第二段内
                if CtrlSys_error_yaw > 0:
                    CtrlSys_temp_sy = kyp1 * CtrlSys_pyth + kyp2 * ( CtrlSys_error_yaw - CtrlSys_pyth )
                else:
                    CtrlSys_temp_sy = -kyp1 * CtrlSys_pyth + kyp2 * ( CtrlSys_error_yaw + CtrlSys_pyth )

            else:
                #--------------------如果在第一段内---------------------------
                CtrlSys_temp_sy = kyp1 * CtrlSys_error_yaw;

            #-------------------计算积分的值----------------------------
            if CtrlSys_error_yaw < CtrlSys_iyth and CtrlSys_error_yaw > -CtrlSys_iyth:
                #-----------计算误差的积分----------------
                CtrlSys_ei2_y = CtrlSys_ei2_y + CtrlSys_error_yaw/10.0
                if CtrlSys_ei2_y>1000:
                    CtrlSys_ei2_y =1000
                if CtrlSys_ei2_y<-1000:
                    CtrlSys_ei2_y = -1000


                CtrlSys_temp_sy = CtrlSys_temp_sy + CtrlSys_ei2_y * kyi / 1000.0
            else:
                #------------超出积分开启的范围，积分无效，并设置误差的积分为0--------
                CtrlSys_ei2_y = 0
            self.CtrlSys_ei2_y = CtrlSys_ei2_y

            #----------------------计算微分的值---------------------
            CtrlSys_cdy =  CtrlSys_error_yaw_last - CtrlSys_error_yaw
            CtrlSys_temp_sy = CtrlSys_temp_sy - CtrlSys_cdy * kyd

            #---------------总比例控制-----------------
            CtrlSys_temp_sy = CtrlSys_temp_sy * kypt

            #--------------输出限位---------------
            if CtrlSys_temp_sy > kylimit:
                CtrlSys_temp_sy = kylimit
            if CtrlSys_temp_sy < -kylimit:
                CtrlSys_temp_sy = -kylimit

            #--------------------输出----------------------
            CtrlSys_sy = CtrlSys_temp_sy

        self.CtrlSys_error_yaw_last = CtrlSys_error_yaw
        return CtrlSys_sy

    def _calActionForPitch(self, pitch, pitchRef):
        CtrlSys_tap = pitchRef * 1800/np.pi
        CtrlSys_cp = pitch * 1800/np.pi
        CtrlSys_ei2_p = self.CtrlSys_ei2_p
        CtrlSys_ppth = self.conf.CtrlSys_ppth
        CtrlSys_ipth = self.conf.CtrlSys_ipth
        CtrlSys_error_pit_last = self.CtrlSys_error_pit_last

        kpp1 = self.conf.kpp1
        kpp2 = self.conf.kpp2
        kpd = self.conf.kpd
        kpi = self.conf.kpi
        kppt = self.conf.kppt
        kpth = self.conf.kpth
        kplimit = self.conf.kplimit
        ##---------------- 俯仰角控制
        #-------------------------俯仰角误差------------------------------%
        if CtrlSys_tap<0:
            CtrlSys_tap = CtrlSys_tap + 3600
        if CtrlSys_tap>3600:
            CtrlSys_tap = CtrlSys_tap-3600

        if CtrlSys_cp<0:
            CtrlSys_cp = CtrlSys_cp + 3600
        if CtrlSys_cp>3600:
            CtrlSys_cp = CtrlSys_cp-3600

        CtrlSys_error_pit = CtrlSys_tap - CtrlSys_cp

        #-------------------如果没有超过门限值，则方向舵角为0-----------------%
        if CtrlSys_error_pit < kpth * 10.0 and CtrlSys_error_pit > -kpth*10.0:
            CtrlSys_temp_sp = 0
            CtrlSys_ei2_p= 0

            #输出
            CtrlSys_sp = CtrlSys_temp_sp

        #-------------------如果超过门限值，则采用PID算法计算输出值-----------------%
        else:
            if CtrlSys_error_pit > 1800:
                #ay 在cy 的左侧e为负,例ay=3300,CtrlSys_cy=300
                CtrlSys_error_pit = -(3600 - CtrlSys_error_pit)
            if CtrlSys_error_pit < -1800:
                #ay在cy的右侧e为正,例ay =300,CtrlSys_cy=3300
                CtrlSys_error_pit = 3600 + CtrlSys_error_pit

            #----------------------计算比例的值------------------------
            if CtrlSys_error_pit > CtrlSys_ppth or CtrlSys_error_pit < -CtrlSys_ppth:
            #如果在第二段内
                if CtrlSys_error_pit > 0:
                    CtrlSys_temp_sp = kpp1 * CtrlSys_ppth + kpp2 * ( CtrlSys_error_pit - CtrlSys_ppth )
                else:
                    CtrlSys_temp_sp = -kpp1 * CtrlSys_ppth + kpp2 * ( CtrlSys_error_pit + CtrlSys_ppth )
            else:
            #--------------------如果在第一段内---------------------------
                CtrlSys_temp_sp = kpp1 * CtrlSys_error_pit

            #-------------------计算积分的值----------------------------
            if CtrlSys_error_pit < CtrlSys_ipth and CtrlSys_error_pit > -CtrlSys_ipth:
                #-----------计算误差的积分----------------
                CtrlSys_ei2_p = CtrlSys_ei2_p + CtrlSys_error_pit/10.0
                if CtrlSys_ei2_p>1000:
                    CtrlSys_ei2_p =1000
                if CtrlSys_ei2_p<-1000:
                    CtrlSys_ei2_p = -1000

                CtrlSys_temp_sp = CtrlSys_temp_sp + CtrlSys_ei2_p * kpi / 1000.0
            else:
            #------------超出积分开启的范围，积分无效，并设置误差的积分为0--------
                CtrlSys_ei2_p = 0
            self.CtrlSys_ei2_p = CtrlSys_ei2_p
            #----------------------计算微分的值---------------------
            CtrlSys_cdp =  CtrlSys_error_pit_last - CtrlSys_error_pit
            CtrlSys_temp_sp = CtrlSys_temp_sp - CtrlSys_cdp * kpd

            #---------------总比例控制-----------------
            CtrlSys_temp_sp = CtrlSys_temp_sp * kppt

            #--------------输出限位---------------
            if CtrlSys_temp_sp > kplimit:
                CtrlSys_temp_sp = kplimit
            if CtrlSys_temp_sp < -kplimit:
                CtrlSys_temp_sp = -kplimit

            #--------------------输出----------------------
            CtrlSys_sp = CtrlSys_temp_sp
        self.CtrlSys_error_pit_last=CtrlSys_error_pit
        return CtrlSys_sp

    def _calActionForRoll(self, roll, rollRef):
        CtrlSys_tar = rollRef * 1800/np.pi
        CtrlSys_cr = roll * 1800/np.pi
        CtrlSys_ei2_r = self.CtrlSys_ei2_r
        CtrlSys_prth = self.conf.CtrlSys_prth
        CtrlSys_irth = self.conf.CtrlSys_irth
        CtrlSys_error_roll_last = self.CtrlSys_error_roll_last

        krp1 = self.conf.krp1
        krp2 = self.conf.krp2
        krd = self.conf.krd
        kri = self.conf.kri
        krpt = self.conf.krpt
        krth = self.conf.krth
        krlimit = self.conf.krlimit
        ##----------------------滚转角控制
        #-------------------------滚转角误差------------------------------%
        if CtrlSys_tar<0:
            CtrlSys_tar = CtrlSys_tar + 3600
        if CtrlSys_tar>3600:
            CtrlSys_tar = CtrlSys_tar-3600

        if CtrlSys_cr<0:
            CtrlSys_cr = CtrlSys_cr + 3600
        if CtrlSys_cr>3600:
            CtrlSys_cr = CtrlSys_cr-3600

        CtrlSys_error_roll = CtrlSys_tar - CtrlSys_cr

        #-------------------如果没有超过门限值，则方向舵角为0-----------------%
        if CtrlSys_error_roll < krth * 10.0 and CtrlSys_error_roll > -krth*10.0:
            CtrlSys_temp_sr = 0
            CtrlSys_ei2_r= 0

            #输出
            CtrlSys_sr = CtrlSys_temp_sr

        #-------------------如果超过门限值，则采用PID算法计算输出值-----------------%
        else:
            if CtrlSys_error_roll > 1800:
            #ay 在cy 的左侧e为负,例ay=3300,CtrlSys_cy=300
                CtrlSys_error_roll = -(3600 - CtrlSys_error_roll)
            if CtrlSys_error_roll < -1800:
            #ay在cy的右侧e为正,例ay =300,CtrlSys_cy=3300
                CtrlSys_error_roll = 3600 + CtrlSys_error_roll

            #---------------------计算比例的值------------------------
            if CtrlSys_error_roll > CtrlSys_prth or CtrlSys_error_roll < -CtrlSys_prth:
                #如果在第二段内
                if CtrlSys_error_roll > 0:
                    CtrlSys_temp_sr = krp1 * CtrlSys_prth + krp2 * ( CtrlSys_error_roll - CtrlSys_prth )
                else:
                    CtrlSys_temp_sr = -krp1 * CtrlSys_prth + krp2 * ( CtrlSys_error_roll + CtrlSys_prth )
            else:
            #--------------------如果在第一段内---------------------------
                CtrlSys_temp_sr = krp1 * CtrlSys_error_roll

            #-------------------计算积分的值----------------------------
            if CtrlSys_error_roll < CtrlSys_irth and CtrlSys_error_roll > -CtrlSys_irth:
            #-----------计算误差的积分---------------
                CtrlSys_ei2_r = CtrlSys_ei2_r + CtrlSys_error_roll/10.0
                if CtrlSys_ei2_r>1000:
                    CtrlSys_ei2_r =1000
                if CtrlSys_ei2_r<-1000:
                    CtrlSys_ei2_r = -1000

                CtrlSys_temp_sr = CtrlSys_temp_sr + CtrlSys_ei2_r * kri / 1000.0
            else:
            #------------超出积分开启的范围，积分无效，并设置误差的积分为0--------
                CtrlSys_ei2_r = 0;
            self.CtrlSys_ei2_r = CtrlSys_ei2_r
            #----------------------计算微分的值---------------------
            CtrlSys_cdr =  CtrlSys_error_roll_last - CtrlSys_error_roll
            CtrlSys_temp_sr = CtrlSys_temp_sr - CtrlSys_cdr * krd

            #---------------总比例控制-----------------
            CtrlSys_temp_sr = CtrlSys_temp_sr * krpt

            #--------------输出限位---------------
            if CtrlSys_temp_sr > krlimit:
                CtrlSys_temp_sr = krlimit
            if CtrlSys_temp_sr < -krlimit:
                CtrlSys_temp_sr = -krlimit

            #--------------------输出----------------------
            CtrlSys_sr = CtrlSys_temp_sr
        self.CtrlSys_error_roll_last = CtrlSys_error_roll
        return CtrlSys_sr

    def calThro(self, u, uRef, yaw, yawRef, pitch,pitchRef, roll,rollRef):

        #AirscrewNCtrl = np.array(self.conf.AirscrewNCtrl) + self._calActionForVel(u, uRef)
        #print(f"u:{u}")
        #print(f"uRef:{uRef}")
        AirscrewNCtrl = np.array(self.conf.AirscrewNCtrl)*(1+uRef)
        CtrlSys_sy = self._calActionForYaw(yaw, yawRef)
        #print("yaw:",yaw)
        #print("yawRef:",yawRef)
        CtrlSys_sp = self._calActionForPitch(pitch, pitchRef)
        CtrlSys_sr = self._calActionForRoll(roll, rollRef)

        CtrlSys=np.array([CtrlSys_sr,CtrlSys_sp,CtrlSys_sy])
        ProCtrlPara  = np.dot(self.Proller_front_AM,np.diag(self.conf.CtrlAllocatePara))
        #print("CtrlSys:",CtrlSys)
        #print("ProCtrlPara:",ProCtrlPara)
        CtrlSys_Thro = AirscrewNCtrl+np.dot(ProCtrlPara,CtrlSys)

        #print("AirscrewNCtrl:",AirscrewNCtrl)
        CtrlSys_Thro = self._ActionChangeLimit(CtrlSys_Thro, self.CtrlSys_Thro_last, self.conf.Thro_Change_Limit)
        CtrlSys_Thro = self._ActionLimit(CtrlSys_Thro, self.CtrlSys_Pro_Max, self.CtrlSys_Pro_Min)
        print("CtrlSys_Thro:",CtrlSys_Thro)
        self.CtrlSys_Thro_last = CtrlSys_Thro
        return CtrlSys_Thro
