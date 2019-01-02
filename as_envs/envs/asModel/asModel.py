from .airConds.asAirConds import AsAirConds
from .config.Config import Config
import json
import math
import numpy as np

class ProModel(object):
    def __init__(self, path="proConf.json"):
        self.conf = Config(path)
    def calU(self, CtrlSys_Thro):
        k_FT = self.conf.k_FT[0]/self.conf.k_FT[1]**2
        Yt = self.conf.Yt
        F = k_FT*(CtrlSys_Thro**2).reshape(2,2)*np.array([[1,0.8],[1,0.8]])
        F = np.sum(F, axis=1)
        #u = np.array([F[0]+F[1],0,0,0,0,Yt*F[0]-Yt*F[1]])
        u = np.array([0,0,0,0,0,0])
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
    def _calvel(self, u,v,w,W,R1):
        Wd = np.dot(R1.T, W.T)
        return np.array([u-Wd[0], v-Wd[1], w])

    def _calAirForce(self, vel, alpha, beta, p, q, r):
        '''
        vel:
        alpha: unit is rad
        beta:unit is read
        p,q,r:
        '''
        alpha = alpha*180.0/math.pi
        beta = beta*180.0/math.pi

        Q = 0.5*self.conf.rho*math.pow(vel,2)
        #print(f'lmn:{self.asCl.getValue(alpha, beta)}')
        asCxyz = np.array([self.asCx.getValue(alpha, beta)[0], \
                           self.asCy.getValue(alpha, beta)[0]*np.sign(beta), \
                           self.asCz.getValue(alpha, beta)[0]*np.sign(alpha)])
        asClmn = np.array([self.asCl.getValue(alpha, beta)[0],\
                           self.asCm.getValue(alpha, beta)[0]*np.sign(alpha), \
                           self.asCn.getValue(alpha, beta)[0]*np.sign(beta)])
        asCpqr = np.array([self.conf.Cp,self.conf.Cq,self.conf.Cr])
        #asCxyz = np.array([0.1,0.03,0.03])
        #asClmn = np.array([0.3,0.3,0.5])

        aspqr = np.array([p,q,r])
        #print(f'alpha:{alpha}')
        #print(f'beta:{beta}')
        #print(f'asCpqr:{asCpqr}')
        #print(f'asCxyz:{asCxyz}')
        #print(f'asClmn:{asClmn}')
        Fa = -Q*asCxyz*self.conf.Sref
        Ma = -Q*(asClmn+asCpqr*aspqr)*self.conf.Sref*self.conf.Lref

        return Fa, Ma

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

    def _calMg(self, theta, phi):
        stheta = math.sin(theta)
        ctheta = math.cos(theta)
        sphi = math.sin(phi)
        cphi = math.cos(phi)
        return np.array([-self.conf.Zc*self.conf.m*self.conf.g*ctheta*sphi+self.conf.Yc*self.conf.m*self.conf.g*ctheta*cphi,
                        -self.conf.Zc*self.conf.m*self.conf.g*stheta-self.conf.Xc*self.conf.m*self.conf.g*ctheta*cphi,
                        self.conf.Xc*self.conf.m*self.conf.g*ctheta*sphi+self.conf.Yc*self.conf.m*self.conf.g*stheta])
    def _calR1(self, phi, theta, psi):
        stheta = math.sin(theta)
        ctheta = math.cos(theta)
        spsi = math.sin(psi)
        cpsi = math.cos(psi)
        sphi = math.sin(phi)
        cphi = math.cos(phi)
        #print(f'phi:{phi}')
        #print(f'theta:{theta}')
        #print(f'psi:{psi}')
        #R1 = np.array([[math.cos(theta)*math.cos(psi),math.sin(theta)*math.cos(psi)*math.sin(phi)-math.sin(psi)*math.cos(phi),math.sin(theta)*math.cos(psi)*math.cos(phi)+math.sin(psi)*math.sin(phi)],\
        #[math.cos(theta)*math.sin(psi),math.sin(theta)*math.sin(psi)*math.sin(phi)+math.cos(psi)*math.cos(phi),math.sin(theta)*math.sin(psi)*math.cos(phi)-math.cos(psi)*math.sin(phi)],\
        #[math.sin(theta),-math.cos(theta)*math.sin(phi),-math.cos(theta)*math.cos(phi)]]);
        return np.array([[ctheta*cpsi, stheta*cpsi*sphi-spsi*cphi, stheta*cpsi*cphi+spsi*sphi],
                        [ctheta*spsi, stheta*spsi*sphi+cpsi*cphi, stheta*spsi*cphi-ctheta*cphi],
                        [stheta, -ctheta*sphi, -ctheta*cphi]])
        return R1
    def _calK1(self, phi, theta):
        stheta = math.sin(theta)
        ctheta = math.cos(theta)
        sphi = math.sin(phi)
        cphi = math.cos(phi)

        return np.array([[1, sphi*stheta/ctheta, cphi*stheta/ctheta],[0, cphi, -sphi],[0, sphi/ctheta, cphi/ctheta]])

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
        #print(f"den1:{den1}")
        #print(f"den2:{den2}")
        addF = self._calAddForce()
        #print(f"addF:{addF}")
        Ix = self.conf.Ix
        Iy = self.conf.Iy
        Iz = self.conf.Iz
        Zc = self.conf.Zc
        m = self.conf.m
        b_11 = (Iy + addF[4])/den2
        b_15 = -m*Zc/den2
        b_22 = (Ix + addF[3]) /den1
        b_24 = m*Zc/den1
        b_33 = -1/(m + addF[2])
        b_42 = m*Zc/den2
        b_44 = (m + addF[1] )/den2
        b_51 = m*Zc/den1
        b_55 = (m + addF[0])/den1
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
        #print(f'addF:{addF}')
        #print(f'Fa:{Fa}')
        #print(f'Ma:{Ma}')
        #print(f'Mg:{Mg}')

        Ix = self.conf.Ix
        Iy = self.conf.Iy
        Iz = self.conf.Iz
        Zc = self.conf.Zc
        m = self.conf.m

        G = self.conf.g*m
        B = G-self.conf.kb*h
        #print(f"G:{G}")
        #print(f"B:{B}")
        ctheta = math.cos(theta)
        stheta = math.sin(theta)
        sphi = math.sin(phi)
        cphi = math.cos(phi)



        F21 = (-(Iz + Iy + addF[4] - Ix)*m*Zc*p*r - (m*Iy + m*addF[4] - m**2*Zc**2)*\
            (w*q - v*r) - m*Zc*(Ma[1] + Mg[1]) + (Iy + addF[4])*Fa[0] - (G - B)*(Iy + addF[4])*\
            stheta)/den2

        F22 = (-(Iz + Ix + addF[3] - Iy )*m*Zc*q*r + (m*Ix + m*addF[3] - m**2*Zc**2)*\
            (w*p - u*r) + m*Zc*(Ma[0] + Mg[0]) + (Ix + addF[3])*Fa[1] + (G - B)*(Ix + addF[3])*\
            ctheta*sphi)/den1

        F23 = (m*(u*q - v*p) + m*Zc*(p**2 + q**2) + Fa[2] + (G - B)*ctheta*cphi)/(m + addF[2])

        F24 = ((m**2*Zc**2 - (m + addF[1])*(Iz - Iy))*q*r - (m + addF[1])*(w*p - u*r) +\
                (m + addF[1] )*(Ma[0] + Mg[0]) + m*Zc*Fa[1] + m*Zc*(G - B)*ctheta*sphi)/den1

        F25 = ((m**2*Zc**2 - (m + addF[0])*(Ix - Iz))*p*r - (m + addF[0]) *(w*q - v*r) +\
                (m + addF[0])*(Ma[1] + Mg[1]) - m*Zc*Fa[0] + m*Zc*(G - B)*stheta)/den2

        F26 = (-(Iy - Ix)*p*q + (Ma[2] + Mg[2]))/(Iz + addF[5])
        #print(f'F21:{F21}')
        #print(f'F22:{F22}')
        #print(f'F23:{F23}')
        #print(f'F24:{F24}')
        #print(f'F25:{F25}')
        #print(f'F26:{F26}')
        return np.array([F21, F22, F23, F24, F25, F26])

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
        #print(f'alpha:{alpha}')
        #print(f'u:{u}')
        #print(f'v:{v}')
        #print(f'w:{w}')
        beta = math.atan2(v*math.cos(alpha),u)

        return alpha, beta


    def calDx(self, X, U, W):
        '''
        X: statues' variables
        U: actions' variables
        W: wind speed relative with I Frame
        '''
        #print(f'X:{X}')
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

        R1 = self._calR1(phi, theta, psi)
        K1 = self._calK1(phi, theta)
        B1 = self._calB1()
        vel = self._calvel(u,v,w,W,R1)
        #print(f'R1:{R1}')
        #print(f'vel:{vel}')
        alpha, beta = self._calTriangle(vel)
        F2 = self._calF2(theta,phi,h,np.linalg.norm(vel),u,v,w, alpha, beta, p, q, r)

        dx1 = np.dot(R1,np.array([u,v,w]).T)
        dx2 = np.dot(K1,np.array([p,q,r]).T)
        dx3 = F2 +np.dot(B1, U.T)
        #print(f'F2:{F2}')
        #print(f'B1:{B1}')
        #print(f'U:{np.dot(B1,U.T)}')
        #print(f'dx1:{dx1}')
        #print(f'dx2:{dx2}')
        #print(f'dx3:{dx3}')
        dx = np.concatenate((dx1.T, dx2.T, dx3.T), axis=0)
        return dx, alpha, beta
