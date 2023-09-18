# -*- coding: utf-8 -*-
"""
@author: victoire
Imperial College London - MSc 2022 - Biomedical engineering
Individual project - Hyperspectral images analysis for neurosurgery

classes.py
in this file:
1) useful function to open extinction coefficient data
2) classes to describe:
    - characteristics of biological tissue (parameters)
    - behaviour of photon packets in biological tissue (photon)
"""

# %% libraries

from math import asin, log, copysign, sqrt, acos, pi, sin, cos
import numpy as np
from random import random

# %% open extinction coefficients data

def Hb_func(LBD):
    sp_data = np.loadtxt('extinction_coefs/blood.txt',skiprows=2,dtype=float)
    LBD1 = sp_data[:,0]                  # wavelength (nm)
    Hb = sp_data[:,2]                    # extinction coef 
    return np.interp(LBD,LBD1,Hb)

def HbO2_func(LBD):
    sp_data = np.loadtxt('extinction_coefs/blood.txt',skiprows=2,dtype=float)
    LBD1 = sp_data[:,0]                  # wavelength (nm)
    Hb02 = sp_data[:,1]                  # extinction coef 
    return np.interp(LBD,LBD1,Hb02)

def H2O_func(LBD):
    sp_data = np.loadtxt('extinction_coefs/water.txt',skiprows=2,dtype=float)
    LBD2 = sp_data[:,0]                   # wavelength (nm)
    H20 = sp_data[:,1]                    # attenuation coef (1/cm)
    return np.interp(LBD,LBD2,H20)

def fat_func(LBD):
    sp_data = np.loadtxt('extinction_coefs/fat.txt',skiprows=1,dtype=float)
    LBD3 = sp_data[:,0]                   # wavelength (nm)
    fat = sp_data[:,1]                    # attenuation coef (1/cm)
    return np.interp(LBD,LBD3,fat)

# %%

class parameters:
    def __init__(self):
        self.g = 0.8                 # anisotropy
        self.n0 = 1.0                # refractive index ambient medium
        self.n1 = 1.4                  # refractive index dura
        self.alpha_c = asin(self.n0/self.n1)   # critical angle
        
        self.Ct = 60*1e-6
        self.SO2 = 0.80
        self.pH2O = 0
        
        self.CHbO2 = self.SO2*self.Ct
        self.CHb = self.Ct - self.CHbO2
        
        self.a = 24
        self.b = 1.6
    
    def mu_a(self,LBD):
        """
        Calculates the absorption coefficient of the blood from its composition

        Parameters
        ----------
        LBD : TYPE : np
            DESCRIPTION : array of wavelengths
        CHbO2 : TYPE : float
            DESCRIPTION : oxyheamoglobin concentration in mol/L
        CHb : TYPE : float
            DESCRIPTION : deoxyheamoglobin concentration in mol/L
        pH2O : TYPE : float
            DESCRIPTION : water percentage

        Returns
        -------
        TYPE : np
            DESCRIPTION : absorption coefficient
        """
        return self.CHbO2*HbO2_func(LBD) + self.CHb*Hb_func(LBD) + self.pH2O*H2O_func(LBD)
    
    def mu_s(self,LBD):
        """
        Calculates scattering coefficient of the blood

        Parameters
        ----------
        LBD : TYPE : np
            DESCRIPTION : array of wavelengths
        a : TYPE : float
            DESCRIPTION : scattering coefficient at lambda = 500 nm (in cm-1)
        b : TYPE
            DESCRIPTION : scattering power (dimentionless)

        Returns
        -------
        TYPE : np
            DESCRIPTION : scattering coefficient (in cm-1)

        """
        return self.a*(LBD/500)**(-self.b)
        
    def new_param(self,Ct_n,SO2_n,a_n,b_n):
        self.Ct = Ct_n
        self.SO2 = SO2_n
        self.CHbO2 = self.SO2*self.Ct
        self.CHb = self.Ct - self.CHbO2
        self.a = a_n
        self.b = b_n

class Photon:
    
    def __init__(self,params):
        
        n0 = params.n0
        n1 = params.n1
        
        # positions
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.pos_memory = np.asarray([[self.x,self.y,self.z]])
        
        # directions
        self.dpx = 0.0
        self.dpy = 0.0
        self.dpz = -1.0
        
        Rsp = ((n0-n1)/(n0+n1))**2 # specular reflectance at first interface
        
        # weight of the photon packet
        self.w = 1.0-Rsp
        
        # distance travelled
        self.D = 0
        
        # light transmitted back to ambient medium
        self.back = False
        
    # def plot_pos(self):
    #     ax1.scatter(self.x,self.z,color='k',s=10)
    #     ax2.scatter(self.y,self.z,color='k',s=10)
        
    def random_step(self,params,lbd, mu_a, mu_t):
        
        self.s = -log(random())/mu_t
        
        # update weight
        dw = self.w * mu_a/mu_t
        if (self.w - dw) < 0:
            dw = self.w
        self.w = self.w - dw
        
        # Check boundary and update position and direction
        sbound = abs(self.z/self.dpz) # normalized distance to boundary
        
        if sbound<copysign(self.s,self.dpz): # boundary is reached
            # update position
            self.x = self.x + self.dpx*sbound
            self.y = self.y + self.dpy*sbound
            self.z = self.z + self.dpz*sbound
            self.D = self.D + sbound*sqrt(self.dpx**2+self.dpy**2+self.dpy**2)
            self.pos_memory = np.vstack([self.pos_memory,[self.x,self.y,self.z]])
            
            alpha_i = acos(abs(self.dpz)) # angle of incidence
            if alpha_i > params.alpha_c: # total reflexion
                self.dpz = - self.dpz
                # print('reflexion')
                self.s = self.s-sbound
                
                # upadate position
                self.x = self.x + self.dpx*self.s
                self.y = self.y + self.dpy*self.s
                self.z = self.z + self.dpz*self.s
                self.D = self.D + self.s*sqrt(self.dpx**2+self.dpy**2+self.dpy**2)
                self.pos_memory = np.vstack([self.pos_memory,[self.x,self.y,self.z]])
                                
            else: #transmission
                # alpha_t = asin(n0 * sin(alpha_i) / n1) # angle of transmission
                # epsilon = 1e-9
                # R = 0.5 * ((sin(alpha_i - alpha_t))**2 / ((sin(alpha_i + alpha_t))**2 + epsilon) + (tan(alpha_i - alpha_t))**2 / ((tan(alpha_i + alpha_t))**2 + epsilon))
                self.w = 0
                # print('transmission', R)
                self.back = True
                
        else: # boundary is not reached
        
            # update position
            self.x = self.x + self.dpx*self.s
            self.y = self.y + self.dpy*self.s
            self.z = self.z + self.dpz*self.s
            self.D = self.D + self.s*sqrt(self.dpx**2+self.dpy**2+self.dpy**2)
            self.pos_memory = np.vstack([self.pos_memory,[self.x,self.y,self.z]])
            
            # update direction
            theta = acos((1 + params.g**2 - ((1 - params.g**2) / (1 - params.g + 2 * params.g * random()))**2) / (2 * params.g))
            psi = 2 * pi * random()
            if abs(self.dpz) > 0.9999:
                self.dpx = sin(theta) * cos(psi)
                self.dpy = sin(theta) * sin(psi)
                self.dpz = copysign(cos(theta), self.dpz)
            else:
                dpx_t = sin(theta) * (self.dpx * self.dpz * cos(psi) - self.dpy * sin(psi)) / sqrt(1 - self.dpz**2) + self.dpx * cos(theta)
                dpy_t = sin(theta) * (self.dpy * self.dpz * cos(psi) + self.dpx * sin(psi)) / sqrt(1 - self.dpz**2) + self.dpy * cos(theta)
                dpz_t = -sin(theta) * cos(psi) * sqrt(1 - self.dpz**2) + self.dpz * cos(theta)
                self.dpx = dpx_t
                self.dpy = dpy_t
                self.dpz = dpz_t