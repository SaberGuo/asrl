from .airConds.asAirConds import AsAirConds
from .config.Config import Config
import json
import math
import numpy as np

class ProModel(object):
    def __init__(self, path="proConf.json"):
        self.conf = Config(path)
        self.y_T= self.conf.z_T*math.cos(self.conf.zy_theta/180.0*np.pi)
        self.z_T= self.conf.z_T*math.sin(self.conf.zy_theta/180.0*np.pi)

    def calU(self, CtrlSys_Thro):
        k_FT = self.conf.k_FT[0]/self.conf.k_FT[1]**2
        Yt = self.conf.Yt
        F = k_FT*(CtrlSys_Thro**2)
        u = np.array([F[0]+F[1]+F[2]+F[3],0,0,0,(-F[0]+F[1]-F[2]+F[3])*self.z_T,(F[0]+F[1]-F[2]-F[3])*self.y_T])
        #u = np.array([0,0,0,0,0,0])
        return u


class AsModel(object):
    def __init__(self,path ="modelConf.json"):
        self.conf = Config(path)
        self.asCx = AsAirConds("conds_cx.txt")
        self.asCy = AsAirConds("conds_cy.txt")
        self.asCz = AsAirConds("conds_cz.txt")
        self.asCn = AsAirConds("conds_cn.txt")
        self.asCm = AsAirConds("conds_cm.txt")
        self.asCl = AsAirConds("conds_cl.txt")
    #计算空速
    # u,v,w--体轴系下的速度
    # W--惯性系下的风速
    # Rwg--转换矩阵，体轴系转换到惯性系，Rwg.T为惯性系转换到体轴系下
    def _calvel(self, u,v,w,W,Rwg):
        Wd = np.dot(Rwg.T, W.T)
        return np.array([u-Wd[0], v-Wd[1], w-Wd[2]])
    #计算气动力与气动力矩
    #vel--体轴系下空速
    #alpha,beta(rad)
    #p,q,r--体轴系下角速度(rad/s)
    def _calAirForce(self, vel, alpha, beta, p, q, r):
        '''
        vel:
        alpha: unit is rad
        beta:unit is read
        p,q,r:
        '''
        alpha = alpha*180.0/math.pi
        beta = beta*180.0/math.pi
        #cx = 0.2
        #cy = 0.1
        #cz = 0.01
        #cl = 0.03
        #cm = 0.03
        #cn = 0.03

        Q = 0.5*self.conf.rho*math.pow(vel,2)
        #print('lmn',self.asCl.getValue(alpha, beta))
        #气动力
        #asCxyz = np.array([cx*math.cos(alpha)*math.cos(beta),\
        #                    cy*calpha*sbeta,\
        #                    cz*salpha])

        asCxyz = np.array([self.asCx.getValue(alpha, beta)[0], \
                           self.asCy.getValue(alpha, beta)[0]*np.sign(beta), \
                           self.asCz.getValue(alpha, beta)[0]*np.sign(alpha)])
        #气动力矩
        asClmn = np.array([self.asCl.getValue(alpha, beta)[0]*np.sign(alpha)*np.sign(beta),\
                           self.asCm.getValue(alpha, beta)[0]*np.sign(alpha), \
                           self.asCn.getValue(alpha, beta)[0]*(np.sign(beta))])
        #角速度系数
        asCpqr = np.array([self.conf.Cp,self.conf.Cq,self.conf.Cr])
        #asCxyz = np.array([0.1,0.03,0.03])
        #asClmn = np.array([0.3,0.3,0.5])

        aspqr = np.array([p,q,r])
        #print('alpha:',alpha)
        #print('beta:',beta)
        #print('asCpqr:',asCpqr)
        #print('asCxyz:',asCxyz)
        #print('asClmn:',asClmn)
        Fa = -Q*asCxyz*self.conf.Sref
        #Ma = -Q*(asClmn+asCpqr*aspqr)*self.conf.Sref*self.conf.Lref
        Ma = -Q*(asClmn)*self.conf.Sref*self.conf.Lref
        return Fa, Ma
    #计算附加惯性力
    def _calAddForce(self):
        a = 0.5*(self.conf.a1+self.conf.a2)
        b = self.conf.b
        e = math.sqrt(1-(self.conf.b**2/a**2))
        f = math.log((1+e)/(1-e))
        g = (1-e**2)/e**3
        alpha = 2*g*(0.5*f-e)
        beta = 1/(e**2)-0.5*g*f
        k1 = alpha/(2-alpha)
        k2 = beta/(2-beta)
        k3 = 0.2*(b**2-a**2)**2*(alpha-beta)/(2*(b**2-a**2)+(b**2+a**2)*(beta-alpha))
        rho = self.conf.rho
        v = self.conf.volume

        return np.array([rho*v*k1, 1.2*rho*v*k2, rho*v*k2, 0, rho*v*k3, 1.2*rho*v*k3])
    #计算重力矩
    def _calMg(self, theta, phi):
        stheta,ctheta = self._fixerrorTriangle(theta)
        sphi,cphi = self._fixerrorTriangle(phi)
        Zc = self.conf.Zc
        m = self.conf.m
        g = self.conf.g
        Xc = self.conf.Xc
        Yc = self.conf.Yc
        return np.array([-Zc*m*g*ctheta*sphi+Yc*m*g*ctheta*cphi,
                        -Zc*m*g*stheta-Xc*m*g*ctheta*cphi,
                        Xc*m*g*ctheta*sphi+Yc*m*g*stheta])
    #计算三角函数
    #angle(rad)
    def _fixerrorTriangle(self,angle):
        sangle = math.sin(angle)
        cangle = math.cos(angle)
        if abs(angle-90/180*math.pi)<0.000001:
            sangle = 1
            cangle = 0

        if abs(angle)<0.000001:
            sangle = 0
            cangle = 1

        return sangle, cangle
    #计算转换矩阵，至体轴系转换惯性系
    #phi,theta,psi:滚转角，俯仰角，偏航角(rad)
    def _calRwg(self, phi, theta, psi):
        sphi,cphi = self._fixerrorTriangle(phi)
        spsi,cpsi = self._fixerrorTriangle(psi)
        stheta,ctheta = self._fixerrorTriangle(theta)

        #print('phi:',phi)
        #print('theta:',theta)
        #print('psi:',psi)
        #R1 = np.array([[math.cos(theta)*math.cos(psi),math.sin(theta)*math.cos(psi)*math.sin(phi)-math.sin(psi)*math.cos(phi),math.sin(theta)*math.cos(psi)*math.cos(phi)+math.sin(psi)*math.sin(phi)],\
        #[math.cos(theta)*math.sin(psi),math.sin(theta)*math.sin(psi)*math.sin(phi)+math.cos(psi)*math.cos(phi),math.sin(theta)*math.sin(psi)*math.cos(phi)-math.cos(psi)*math.sin(phi)],\
        #[math.sin(theta),-math.cos(theta)*math.sin(phi),-math.cos(theta)*math.cos(phi)]]);
        return np.array([[ctheta*cpsi, stheta*cpsi*sphi-spsi*cphi, stheta*cpsi*cphi+spsi*sphi],
                        [ctheta*spsi, stheta*spsi*sphi+cpsi*cphi, stheta*spsi*cphi-cpsi*sphi],
                        [-stheta, ctheta*sphi, ctheta*cphi]])
        #return np.array([[ctheta*cpsi, ctheta*spsi, -stheta],
        #                [stheta*cpsi*sphi-spsi*cphi, stheta*spsi*sphi+cpsi*cphi, ctheta*sphi],
        #                [stheta*cpsi*cphi+spsi*sphi, stheta*spsi*cphi-ctheta*cphi, ctheta*cphi]])
        #return R1
    #计算中间量K1
    #phi,theta, 滚转和俯仰角(rad)
    def _calK1(self, phi, theta):

        stheta,ctheta = self._fixerrorTriangle(theta)
        sphi,cphi = self._fixerrorTriangle(phi)

        return np.array([[1, sphi*stheta/ctheta, cphi*stheta/ctheta],[0, cphi, -sphi],[0, sphi/ctheta, cphi/ctheta]])
    #计算中间参数
    def _calDen(self):
        m = self.conf.m
        Ix = self.conf.Ix
        Iy = self.conf.Iy
        Zc = self.conf.Zc
        addF = self._calAddForce()
        den1 = m*Ix + Ix*addF[1]-m**2*Zc**2+m*addF[3]+addF[1]*addF[3]
        den2 = m*Iy + m*addF[4]-m**2*Zc**2+addF[0]*addF[4]+addF[0]*Iy

        return den1, den2

    def _calB1(self):
        den1, den2 = self._calDen()
        #print('den1:',den1)
        #print('den2:',den2)
        addF = self._calAddForce()
        #print('addF:',addF)
        Ix = self.conf.Ix
        Iy = self.conf.Iy
        Iz = self.conf.Iz
        Zc = self.conf.Zc
        m = self.conf.m
        b_11 = (Iy + addF[4])/den2
        b_15 = -m*Zc/den2
        b_22 = (Ix + addF[3]) /den1
        b_24 = m*Zc/den1
        b_33 = 1/(m + addF[2])
        b_42 = m*Zc/den1
        b_44 = (m + addF[1] )/den1
        b_51 = m*Zc/den2
        b_55 = (m + addF[0])/den2
        b_66 = 1/(Iz + addF[5])

        return np.array([[b_11,0,0,0,b_15,0],
        [0,b_22,0,b_24,0,0],
        [0,0,b_33,0,0,0],
        [0,b_42,0,b_44,0,0],
        [b_51,0,0,0,b_55,0],
        [0,0,0,0,0,b_66]])

    def _calF2(self,theta, phi, h, vel, u,v,w, alpha, beta, p, q, r):

        den1, den2 = self._calDen()
        addF = self._calAddForce()
        Fa, Ma = self._calAirForce(vel, alpha, beta, p, q, r)


        Mg = self._calMg(theta, phi)
        #print('addF',addF)
        #print('Fa:',Fa)
        #print('Ma:',Ma)
        #print('Mg:',Mg)

        Ix = self.conf.Ix
        Iy = self.conf.Iy
        Iz = self.conf.Iz
        Zc = self.conf.Zc
        m = self.conf.m

        G = self.conf.g*m
        B = G
        #B = G-self.conf.kb*h
        #print('G:',G)
        #print('B:',B)
        stheta,ctheta = self._fixerrorTriangle(theta)
        sphi,cphi = self._fixerrorTriangle(phi)
        #ctheta = math.cos(theta)
        #stheta = math.sin(theta)
        #sphi = math.sin(phi)
        #cphi = math.cos(phi)



        F21 = (-(Iz + Iy + addF[4] - Ix)*m*Zc*p*r - ((m+addF[0])*(Iy + addF[4]) - m**2*Zc**2)*\
    (w*q - v*r) - m*Zc*(Ma[1] + Mg[1]) + (Iy + addF[4])*Fa[0] - (G - B)*(Iy + addF[4])*\
    stheta)/den2

        F22 = (-(Iz + Ix + addF[3] - Iy )*m*Zc*q*r + ((m+addF[1])*(Ix + addF[3]) - m**2*Zc**2)*\
    (w*p - u*r) + m*Zc*(Ma[0] + Mg[0]) + (Ix + addF[3])*Fa[1] + (G - B)*(Ix + addF[3])*\
    ctheta*sphi)/den1

        F23 = (m*(u*q - v*p) + m*Zc*(p**2 + q**2) + Fa[2] + (G - B)*ctheta*cphi)/(m + addF[2])

        F24 =((m**2*Zc**2 - (m + addF[1])*(Iz - Iy))*q*r +\
    (m + addF[1] )*(Ma[0] + Mg[0]) + m*Zc*Fa[1] + m*Zc*(G - B)*ctheta*\
    sphi)/den1

        F25 =((m**2*Zc**2 - (m + addF[0])*(Ix - Iz))*p*r + \
    (m + addF[0])*(Ma[1] + Mg[1]) - m*Zc*Fa[0] + m*Zc*(G - B )*stheta)/den2

        F26 =(-(Iy - Ix)*p*q + (Ma[2] + Mg[2]))/(Iz + addF[5])
        #print('F21:',F21)
        #print('F22:',F22)
        #print('F23:',F23)
        #print('F24:',F24)
        #print('F25:',F25)
        #print('F26:',F26)
        return np.array([F21, F22, F23, F24, F25, F26])
    #计算 alpha,beta 迎角和侧滑角
    #vel--空速，体轴系下
    #return: alpha,beta(rad)
    def _calTriangle(self, vel):
        u = vel[0]
        v = vel[1]
        w = vel[2]

        alpha = 0
        beta = 0
        if w == 0:
            alpha = 0
        elif u == 0 and w>0:
            alpha = math.pi/2
        elif u == 0 and w<0:
            alpha = -math.pi/2
        else:
            alpha = math.atan(w/u)

        #print('u:',u)
        #print('v:',v)
        #print('w:',w)
        beta = math.atan2(v*math.cos(alpha),u)
        alpha = min(np.abs(alpha), 15*np.pi/180.0)*np.sign(alpha)
        beta = min(np.abs(beta), 45*np.pi/180.0)*np.sign(beta)
        #print('alpha:',alpha*180.0/np.pi)
        #print('beta:',beta*180.0/np.pi)
        return alpha, beta


    def calDx(self, X, U, W):
        '''
        X: statues' variables
        U: actions' variables
        W: wind speed relative with I Frame
        '''
        #print('X:',X)
        x = X[0]
        y = X[1]
        h = X[2]
        phi = X[3]
        theta = X[4]
        psi = X[5]
        u = X[6]
        v= X[7]
        w = X[8]
        p = X[9]
        q = X[10]
        r = X[11]

        Rwg = self._calRwg(phi, theta, psi)
        K1 = self._calK1(phi, theta)
        B1 = self._calB1()
        vel = self._calvel(u,v,w,W,Rwg)
        #print('R1:',R1)
        #print('vel:',vel)
        alpha, beta = self._calTriangle(vel)
        F2 = self._calF2(theta,phi,h,np.linalg.norm(vel),u,v,w, alpha, beta, p, q, r)

        dx1 = np.dot(Rwg,np.array([u,v,w]).T)
        dx2 = np.dot(K1,np.array([p,q,r]).T)
        dx3 = F2 +np.dot(B1, U.T)
        #print('F2:',F2)
        #print('B1:',B1)
        #print('U:',np.dot(B1,U.T))
        #print('dx1',dx1)
        #print('dx2',dx2)
        #print('dx3',dx3)
        dx = np.concatenate((dx1.T, dx2.T, dx3.T), axis=0)
        return dx, alpha, beta
