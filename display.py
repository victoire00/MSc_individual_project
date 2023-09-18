# -*- coding: utf-8 -*-
"""
@author: victoire
Imperial College London - MSc 2022 - Biomedical engineering
Individual project - Hyperspectral images analysis for neurosurgery

display.py
in this file:
functions to display all relevant results of this project
includes report figures and functions for figures aspect
"""

# %% libraries

import numpy as np
import matplotlib.pyplot as plt
from functions import random_walk, launch_N_photons, pathlength_wv
from labels import plot_img_labels
import matplotlib.patches as patches
from time import time
import json
from os import listdir

from classes import parameters, Hb_func, HbO2_func

# %%

def axis00(AX,xlabel,ylabel,fs):
    """
    Function for the aspect of plots of MC simulation

    Parameters
    ----------
    AX : TYPE : matplotlib.pyplot ax
        DESCRIPTION.
    xlabel : TYPE : string
        DESCRIPTION.
    ylabel : TYPE : string
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    AX.set_facecolor((0.97, 0.97, 0.97))
    AX.set_aspect('equal')
    
    # set the x-spine
    AX.spines['left'].set_position('zero')
    
    # turn off the right spine/ticks
    AX.spines['right'].set_color('none')
    AX.yaxis.tick_left()
    
    # set the y-spine
    AX.spines['bottom'].set_position('zero')
    
    # turn off the top spine/ticks
    AX.spines['top'].set_color('none')
    AX.xaxis.tick_bottom()
    
    AX.set_xlabel(xlabel,loc='right',fontsize=fs)
    AX.set_ylabel(ylabel,loc='bottom',fontsize=fs)
    
def wavelength_to_rgb(wavelength):
    """
    calculate a rgb tuple from a wavelength value
    this function was written by chat GPT

    Parameters
    ----------
    wavelength : TYPE : float
        DESCRIPTION : wavelength

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    red : TYPE : float
        DESCRIPTION : red
    green : TYPE : float
        DESCRIPTION : red
    blue : TYPE : float
        DESCRIPTION : red

    """
    if wavelength < 380 or wavelength > 780:
        raise ValueError("Wavelength must be between 380 and 780 nanometers.")
    
    red, green, blue = 0.0, 0.0, 0.0
    
    if wavelength >= 380 and wavelength < 440:
        red = -(wavelength - 440) / (440 - 380)
        green = 0.0
        blue = 1.0
    elif wavelength >= 440 and wavelength < 490:
        red = 0.0
        green = (wavelength - 440) / (490 - 440)
        blue = 1.0
    elif wavelength >= 490 and wavelength < 510:
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510) / (510 - 490)
    elif wavelength >= 510 and wavelength < 580:
        red = (wavelength - 510) / (580 - 510)
        green = 1.0
        blue = 0.0
    elif wavelength >= 580 and wavelength < 645:
        red = 1.0
        green = -(wavelength - 645) / (645 - 580)
        blue = 0.0
    elif wavelength >= 645 and wavelength < 781:
        red = 1.0
        green = 0.0
        blue = 0.0
    
    # Intensity correction
    if wavelength >= 380 and wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength >= 420 and wavelength < 701:
        factor = 1.0
    elif wavelength >= 701 and wavelength < 781:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 0.0
    
    # Apply intensity correction
    red *= factor
    green *= factor
    blue *= factor
    
    return red,green,blue

# %% report

def plot_extinction_coef():
    ymin,ymax = 0,0.6 # graph hight
    text_h = 0.57 # text position
    fs = 13 # fontsize
    LBD = np.linspace(400,800,500)
    
    fig,ax1 = plt.subplots(figsize = (8,5))
    
    ax1.plot(LBD,Hb_func(LBD)*1e-6,label=r'$Hb$',color='b')
    ax1.plot(LBD,HbO2_func(LBD)*1e-6,label=r'$Hb0_2$',color='r')
    rect = patches.Rectangle((440, ymin), 720-440, ymax, linewidth=2, edgecolor='none', facecolor=(0.9,0.9,0.9))
    ax1.add_patch(rect)
    ax1.text((720+440)/2, text_h, 'LCTF spectral range', color='k', fontsize=fs, ha='center', va='center')
    ax1.annotate(' ', xy=(440,text_h-0.015), xytext=(720,text_h-0.02),arrowprops=dict(edgecolor='k', arrowstyle='<->'))
    ax1.set_ylim([ymin,ymax])
    ax1.set_xlabel('wavelength (nm)',fontsize=fs)
    ax1.set_ylabel(r'molar extinction coefficient $\varepsilon(\lambda)$ ($cm^{-1}.\mu mol^{-1}$)',fontsize=fs)
    ax1.legend(loc='center right',fontsize=fs)
    
    plt.show()

def plot_abs_scat_coefs():
    ymin,ymax = 0,37 # graph hight
    text_h = 35 # text position
    fs = 13 # fontsize
    LBD = np.linspace(400,800,500)
    params1 = parameters()
    
    fig,ax1 = plt.subplots(figsize = (8,5))

    ax1.plot(LBD,params1.mu_a(LBD),label=r'$\mu _a$',color='k')
    ax1.plot(LBD,params1.mu_s(LBD),label=r'$\mu _s$',color='g')
    rect = patches.Rectangle((440, ymin), 720-440, ymax, linewidth=2, edgecolor='none', facecolor=(0.9,0.9,0.9))
    ax1.add_patch(rect)
    ax1.text((720+440)/2, text_h, 'LCTF spectral range', color='k', fontsize=fs, ha='center', va='center')
    ax1.annotate(' ', xy=(440,text_h-1.5), xytext=(720,text_h-2),arrowprops=dict(edgecolor='k', arrowstyle='<->'))
    ax1.set_ylim([ymin,ymax])
    ax1.set_xlabel('wavelength (nm)',fontsize=fs)
    ax1.set_ylabel(r'coefficients $\mu _a(\lambda)$ and $\mu _s(\lambda)$ ($cm^{-1}$)',fontsize=fs)
    ax1.legend(loc='center right',fontsize=fs)
    
    plt.show()

# %% monte carlo

def plot_random_walk_1P():
    
    params1 = parameters()
    lbd = 500
    mu_a, mu_s = params1.mu_a(lbd), params1.mu_s(lbd)
    mu_t = mu_a + mu_s
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,4))
    p1 = random_walk(params1,lbd, mu_a, mu_t)
    if p1.back:
        ax1.plot(p1.pos_memory[:,0],p1.pos_memory[:,2],'g')
        ax2.plot(p1.pos_memory[:,1],p1.pos_memory[:,2],'g')
    else:
        ax1.plot(p1.pos_memory[:,0],p1.pos_memory[:,2],'r')
        ax2.plot(p1.pos_memory[:,1],p1.pos_memory[:,2],'r')
    axis00(ax1,'x (cm)','z (cm)')
    axis00(ax2,'y (cm)','z (cm)')
    plt.axis('equal')
    plt.show()

def plot_random_walk_NP():
    fs = 13 # fontsize
    params1 = parameters()
    lbd = 800
    mu_a, mu_s = params1.mu_a(lbd), params1.mu_s(lbd)
    mu_t = mu_a + mu_s
    N = 1000
    
    # fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,4))
    fig, ax1 = plt.subplots(figsize = (8,5))
    dist_memory = []
    posback_memory = []
    for i in range(N):
        p1 = random_walk(params1, lbd, mu_a, mu_t)
        if p1.back:
            dist_memory.append(p1.D)
            posback_memory.append([p1.x,p1.y])
            ax1.plot(p1.pos_memory[:,0],p1.pos_memory[:,2],'g',linewidth=1, alpha=0.25)
            # ax2.plot(p1.pos_memory[:,1],p1.pos_memory[:,2],'g')
        else:
            ax1.plot(p1.pos_memory[:,0],p1.pos_memory[:,2],'r',linewidth=1, alpha=0.25)
            # ax2.plot(p1.pos_memory[:,1],p1.pos_memory[:,2],'r')
    axis00(ax1,'x (cm)','z (cm)',fs)
    # axis00(ax2,'y (cm)','z (cm)')
    plt.show()
    
    
    # fig2, (ax3,ax4) = plt.subplots(1,2,figsize=(13,4))
    fig2, ax3 = plt.subplots(figsize=(8,5))
    posback_memory = np.asarray(posback_memory)
    ax3.scatter(posback_memory[:,0],posback_memory[:,1],s=2,color = 'g')
    axis00(ax3,'x (cm)','y (cm)',fs)
    # ax3.set_xlim([-1,1])
    # ax3.set_ylim([-1,1])
    
    fig2bis, (ax42,ax4) = plt.subplots(2,1,figsize=(8,5), gridspec_kw={'height_ratios': [1,3]})
    dist = np.sqrt(posback_memory[:,0]**2 + posback_memory[:,1]**2)
    ax4.hist(dist,density=False,bins=50, facecolor = '#2088B2', edgecolor='#FFFFFF', linewidth=0.5)
    ax4.set_xlabel('distance from center (cm)', fontsize=fs)
    ax4.set_ylabel('number of photons', fontsize=fs)
    ax42.boxplot(dist, vert=False)
    # ax4.set_ylim([0,100])
    plt.tight_layout()
    plt.show()
    
    print(np.size(dist_memory),'/',N,"photons are transmitted back to the ambient medium")

    fig3, (ax52,ax5) = plt.subplots(2,1,figsize=(8,5), gridspec_kw={'height_ratios': [1,3]})
    ax5.hist(dist_memory,density=False,bins=50, facecolor = '#2088B2', edgecolor='#FFFFFF', linewidth=0.5)
    ax5.set_ylabel('number of photons', fontsize=fs)
    ax5.set_xlabel('distance travelled (cm)',fontsize=fs)
    ax52.boxplot(dist_memory, vert=False)
    plt.tight_layout()
    plt.show()

def plot_pathlength_wv():
    params1 = parameters()
    LBD = np.linspace(400, 800, 500)
    pathlength = pathlength_wv(params1,LBD)
    fig, ax = plt.subplots(figsize=(7,8))
    ax.plot(LBD, pathlength)
    plt.show()


def plot_saved_pathlength():
    fs=13
    fig,ax2 = plt.subplots(figsize = (10,5))
    folder = 'saved_dpf_bis/'
    for json_name in listdir(folder):
        Ct = json.load(open(folder+json_name, 'r'))['Ct']*1e6
        SO2 = json.load(open(folder+json_name, 'r'))['SO2']*100
        LBD = json.load(open(folder+json_name, 'r'))['LBD']
        D = json.load(open(folder+json_name, 'r'))['D']
        # ax1.plot(np.asarray(LBD),np.asarray(D),label = r'$C_t=%.2f\mu mol/L;SO_2=%.2f$' %(Ct,SO2))
        ax2.plot(np.asarray(LBD),np.asarray(D),label = r'$C_t=%.2f\mu mol/L;SO_2=%.2f$' %(Ct,SO2))

    # ax1.set_xlabel('wavelength (nm)',fontsize=fs)
    # ax1.set_ylabel('distance travelled (cm)',fontsize=fs)
    # ax1.legend(fontsize=fs)
    ax2.set_xlabel('wavelength (nm)',fontsize=fs)
    ax2.set_ylabel('distance travelled (cm)',fontsize=fs)
    ax2.legend(fontsize=fs)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


# %% time comparison

def plot_time_launch_NP():
    fs = 13
    params1 = parameters()
    
    # plot 1
    LBD = [450,500,600,625,650,700]
    N_list = [50,200,300,500, 600]
    fig,ax = plt.subplots(figsize=(8,7))
    for lbd in LBD:
        mu_a, mu_s = params1.mu_a(lbd), params1.mu_s(lbd)
        mu_t = mu_a + mu_s
        rgb = wavelength_to_rgb(lbd)
        time_memory = []
        for N in N_list:
            t1 = time()
            launch_N_photons(params1,lbd,N, mu_a, mu_t)
            t2 = time()
            time_memory.append(t2-t1)
        ax.plot(N_list,time_memory,color = rgb, label = str(lbd)+"nm")
    ax.set_xlabel("number of photon launched", fontsize=fs)
    ax.set_ylabel("time (sec)", fontsize=fs)
    plt.legend(fontsize=fs)
    plt.title("Monte-Carlo Simulations, duration of calculation according to wavelength", fontsize=fs)
    plt.show()
    
    # plot 2
    LBD = np.linspace(400,740,101)
    N = 500
    fig,ax = plt.subplots(figsize=(8,7))
    for lbd in LBD:
        mu_a, mu_s = params1.mu_a(lbd), params1.mu_s(lbd)
        mu_t = mu_a + mu_s
        time_memory = []
        

# %%

def plot_labels():
    folder1 = '../../../../Individual project/'
    folder2 = 'Registered data/'
    # folder2 = 'Raw data - do not edit/'
    folder3 = 'HS046/'
    field = 'Field01/'
    file_path = folder1 + folder2 + folder3 + field
    plot_img_labels(file_path)

def plot_SO2_map(SO2_map):
    fs = 13
    # # Define the color map
    # colors = [(0, 0, 1), (1, 0, 0)]  # Blue to Red
    # cmap_name = 'blue_to_red'
    # cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    fig, ax = plt.subplots()
    AX = ax.imshow(SO2_map, cmap='coolwarm', vmin = 0, vmax = 1)
    # AX = ax.imshow(SO2_map, cmap='coolwarm')

    # Add a colorbar
    cbar = plt.colorbar(AX, ax=ax, orientation='vertical')
    cbar.set_label(r'Oxygen Saturation $(SO_2)$', fontsize=fs)


def plot_Ct_map(Ct_map):
    fs = 13
    
    fig, ax = plt.subplots()
    # AX = ax.imshow(Ct_map, cmap='coolwarm', vmin = 14*1e-6, vmax = 44*1e-6)
    AX = ax.imshow(Ct_map, cmap='coolwarm')

    # Add a colorbar
    cbar = plt.colorbar(AX, ax=ax, orientation='vertical')
    cbar.set_label(r'Total Hemoglobin concentration $(C_t)$', fontsize=fs)



def plot_SO2_runtime():
    fs=13
    blocs = [17, 48, 117, 207, 602, 973]
    time = [9.2, 26.3, 55.1, 107, 386, 571]
    fig, ax = plt.subplots(figsize = (8,5))
    
    ax.plot([0,4.3*1000],[0,4.3*597.8], 'k--')
    ax.scatter(blocs,time,color='k',marker='+', s=100)
    prev = np.asarray([1803, 4217])
    ax.scatter(prev, prev*0.5978, color='r', marker='+', s=100)
    
    
    ax.set_xlabel('number of blocs', fontsize=fs)
    ax.set_ylabel('time (sec)', fontsize=fs)
    plt.show()

# %% LCTF characteristics

def gaussian(x, lbd0, fwhm, a0):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # calculate standard deviation
    y = a0* np.exp(-(x - lbd0)**2 / (2 * sigma**2))
    return y

def LCTF_characteristics():
    n=1000
    LBD0 = np.linspace(440,720,29)
    FWHM = [10.03, 11.86, 12.1, 12.09, 12.6, 14.42, 14.14, 15.18, 15.41, 18.01, 19.03, 18.22, 20.02, 20.26, 21.52, 22.54, 24.32, 25.84, 26.58, 26.29, 28.31, 28.79, 26.7, 28.46, 38, 26, 18, 17, 12]
    A0 = [0.0073, 0.0083, 0.0134, 0.0167, 0.0178, 0.0243, 0.0277, 0.0303, 0.031, 0.0327, 0.0331, 0.0351, 0.0331, 0.0349, 0.0336, 0.0349, 0.0385, 0.0368, 0.0358, 0.042, 0.0404, 0.0409, 0.0457, 0.0462, 0.0383, 0.0429, 0.0462, 0.0407, 0.0256 ]
    x = np.linspace(400,750,n)
    data = np.zeros((30,n))
    data[0,:]=x
    fig,ax = plt.subplots()
    for j in range(1,30):
        i=j-1
        lbd0, fwhm, a0 = LBD0[i], FWHM[i], A0[i]
        if i == 24:
            fwhm,a0 = (FWHM[25]+FWHM[23])/2,(A0[25]+A0[23])/2
        y = gaussian(x,lbd0, fwhm, a0)
        rgb = wavelength_to_rgb(lbd0)
        plt.plot(x,y, color = rgb)
        data[i,:]=y
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('transmission')
    plt.show()
    np.save('camera_response/lctf_gaussian.npy',data)