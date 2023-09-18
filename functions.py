# -*- coding: utf-8 -*-
"""
@author: victoire
Imperial College London - MSc 2022 - Biomedical engineering
Individual project - Hyperspectral images analysis for neurosurgery

functions.py
in this code:
1) all the functions for the MC simulation
2) all the functions for the model fitting
"""

# %% libraries

import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
import cv2
import json
import time
from classes import Photon, parameters
from scipy.optimize import curve_fit
from PIL import Image, ImageDraw

# %%

def openimg(img_path):
    """
    Create a numpy matrix containing the data of HS image 

    Parameters
    ----------
    img_path : TYPE : str
        DESCRIPTION : HS image path

    Returns
    -------
    img1 : TYPE : np
        DESCRIPTION : h x w x 29 matrix

    """
    print("--Opening image--")
    t1 = time.time()
    img_names = os.listdir(img_path)
    img1 = cv2.imread(img_path+img_names[1], cv2.IMREAD_UNCHANGED)
    img1 = img1[:, :, None]
    h,w,_ = np.shape(img1)
    nb_avg = 1
    for k in range(2, len(img_names)):
        img2 = cv2.imread(img_path + img_names[k], cv2.IMREAD_UNCHANGED)
        img2 = img2[:, :, None]
        img1 = np.concatenate((img1, img2), axis=-1)
    if np.shape(img1)[-1]==58:
        nb_avg = 2
        reshaped_matrix = np.reshape(img1, (h, w, 29, nb_avg))
        img1 = np.sum(reshaped_matrix, axis=3)/nb_avg
    if np.shape(img1)[-1]==116:
        nb_avg = 4
        reshaped_matrix = np.reshape(img1, (h, w, 29, nb_avg))
        img1 = np.sum(reshaped_matrix, axis=3)/nb_avg
    t2 = time.time()
    print("Duration :",t2-t1,"sec")
    print("--Image info--")
    print("HS image shape :",h,w,29)
    print("Number of averages :",nb_avg)
    return img1


# %% Monte carlo

def random_walk(params,lbd, mu_a, mu_t):
    """
    this function function makes the photon take random steps until its weight is considered zero

    Parameters
    ----------
    params : TYPE : parameters object
        DESCRIPTION : biological tissue characteristics
    lbd : TYPE : float
        DESCRIPTION : photon wavelength (nm)
    mu_a : TYPE : float
        DESCRIPTION : absorption coefficient
    mu_t : TYPE : float
        DESCRIPTION : total attenuation coefficient

    Returns
    -------
    p1 : TYPE
        DESCRIPTION.

    """
    p1 = Photon(params)
    while p1.w>0.001:
        p1.random_step(params,lbd, mu_a, mu_t)
    return p1

def launch_N_photons(params,lbd,N, mu_a, mu_t):
    print("---Launching",N,"photons--")
    t1 = time.time()
    dist_memory = []
    for i in range(N):
        p1 = random_walk(params, lbd, mu_a, mu_t)
        if p1.back == True:
            dist_memory.append(p1.D)
    t2 = time.time()
    print("---Duration :",t2-t1, "sec")
    return np.asarray(dist_memory)

def smooth_data(array):
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(array, kernel, mode='same')

def pathlength_wv(params,LBD):
    print("-MC simulation for all wavelengths--")
    t1 = time.time()
    mean_memory = []
    for lbd in LBD:
        mu_a, mu_s = params.mu_a(lbd), params.mu_s(lbd)
        mu_t = mu_a + mu_s
        print('--lbd =',lbd)
        if lbd<550:
            N = 200
        if lbd>=550:
            N = 500
        dist_memory = launch_N_photons(params,lbd,N, mu_a, mu_t)
        mean_memory.append(np.mean(dist_memory))
    t2 = time.time()
    print("-Total duration :",t2-t1)
    return smooth_data(np.asarray(mean_memory))
    
def filename_func(t):
    return str(t.tm_year)+str(t.tm_mon)+str(t.tm_mday)+'_'+str(t.tm_hour)+'h'+str(t.tm_min)+'min'+str(t.tm_sec)

def save_pathlength(params,LBD,array):
    filename = filename_func(time.localtime())
    data = {
        "n0": params.n0,
        "n1": params.n1,
        "g": params.g,
        "Ct": params.Ct,
        "SO2": params.SO2,
        "pH2O": params.pH2O,
        "a": params.a,
        "b": params.b,
        "LBD": LBD[3:-3].tolist(),
        "D": array[3:-3].tolist()
        }
    # Serializing json
    json_object = json.dumps(data, indent=10)
    # Writing to sample.json
    folder = "saved_dpf/"
    with open(folder+filename+".json", "w") as outfile:
        outfile.write(json_object)

def save_diff_comp(params,LBD):
    for Ct_n in [30*1e-6,50*1e-6,70*1e-6,100*1e-6]:
        for SO2_n in [0.8]:
            for a_n in [24]:
                for b_n in [1.6]:
                    print('Ct=',Ct_n,'SO2=',SO2_n,"a_n=",a_n,"b_n=",b_n)
                    params.new_param(Ct_n, SO2_n, a_n, b_n)
                    array = pathlength_wv(params,LBD)
                    save_pathlength(params,LBD,array)

# %%

def open_dpf(params,LBD):
    folder = "saved_dpf/"
    list_name = [json_name for json_name in listdir(folder)]
    i=0
    Ct = params.Ct
    SO2 = params.SO2
    Ct_ = json.load(open(folder+list_name[i], 'r'))['Ct']
    SO2_ = json.load(open(folder+list_name[i], 'r'))['SO2']
    while (Ct_,SO2_) != (Ct,SO2):
        i+=1
        Ct_ = json.load(open(folder+list_name[i], 'r'))['Ct']
        SO2_ = json.load(open(folder+list_name[i], 'r'))['SO2']
    D = np.asarray(json.load(open(folder+list_name[i], 'r'))['D'])
    LBD1 = np.asarray(json.load(open(folder+list_name[i], 'r'))['LBD'])
    
    return np.interp(LBD,LBD1,D)

def I_func(LBD, I0, Ct_n, SO2_n, params, dpf):
    params.new_param(Ct_n, SO2_n, 24, 1.6)
    A = params.mu_a(LBD)*dpf
    return I0*10**(-A)

def fit_I(LBD,I):
    params = parameters()
    dpf = open_dpf(params,LBD)
    I_func2 = lambda LBD, I0, Ct_n, SO2_n: I_func(LBD, I0, Ct_n, SO2_n, params, dpf)
    popt, pcov = curve_fit(I_func2, LBD, I ,bounds=([0.1,5*1e-6,0.4],[10,110*1e-6,1]))
    return popt,I_func2(LBD, *popt)

def open_camera_response(LBD,cam = 'cam2'):
    folder = "camera_response/"
    LBD1 = np.asarray(json.load(open(folder+'wr_'+cam+'.json', 'r'))['lbd'])
    CR = np.asarray(json.load(open(folder+'wr_'+cam+'.json', 'r'))['spectrum'])
    return np.interp(LBD,LBD1,CR)

def onclick(event,img):
    fs=13
    ix, iy = int(event.xdata), int(event.ydata)
    LBD = np.linspace(440,720,29)
    
    # --> if camera 2 is used
    I = np.asarray(img[iy,ix,:])/65536
    ref = open_camera_response(LBD,'cam2')
    # --> if camera 1 is used
    # I = np.asarray(img[iy,ix,:])/255
    # ref = open_camera_response(LBD,'cam1')
    
    I = I/ref
    
    popt,I_fit = fit_I(LBD,I)
    print("I0 =",popt[0])
    print("Ct =",int(1e8*popt[1])/1e8,"mol")
    print("SO2 =",int(1000*popt[2])/10,"%")
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(LBD,I, label='pixel intensity')
    ax.plot(LBD,I_fit, label='fitted data')
    ax.set_xlabel('wavelength (nm)',fontsize = fs)
    plt.legend(fontsize=fs)
    plt.show()
    
    fig2,ax2 = plt.subplots(figsize = (5,5))
    ax2.imshow(img[:,:,19],cmap='gray')
    ax2.scatter([ix],[iy],marker='+',color='r')
    plt.show()
    

def ask_click(img,title):
    fig,ax = plt.subplots(figsize = (10,10))
    ax.imshow(img[:,:,19],cmap='gray')
    plt.title('click on any pixel - '+title)
    callback = lambda event: onclick(event, img)
    cid = fig.canvas.mpl_connect('button_press_event', callback)
    plt.tight_layout()

# %% mask for segmented tissue

def create_mask(w,h,polygon):
    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).polygon(polygon.tolist(), outline=1, fill=1)
    mask = np.array(img)
    return mask

def saveclick(event,file_path):
    file_name = file_path+'/polygon.npy'
    ix, iy = int(event.xdata), int(event.ydata)
    try:
        data = np.load(file_name)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        data = np.vstack((data, [ix,iy]))
    except FileNotFoundError:
        data = np.array([ix,iy])
    np.save(file_name, data)

def draw_polygon(img,file_path):
    fig,ax = plt.subplots(figsize = (10,10))
    ax.imshow(img[:,:,19],cmap='gray')
    callback = lambda event: saveclick(event, file_path)
    cid = fig.canvas.mpl_connect('button_press_event', callback)
    plt.tight_layout()
    plt.show()

def seg_polygon(file_path,tissue):
    """
    this function creates a mask of the desired tissue using the segmentation from the specialist

    Parameters
    ----------
    file_path : TYPE : string
        DESCRIPTION : file path
    tissue : TYPE : string
        DESCRIPTION : desired tissue, example : 'Dura' 

    Returns
    -------
    mask : TYPE : matrix of 0 and 1
        DESCRIPTION : mask

    """
    print("--Creating mask--")
    t1 = time.time()
    file_name = 0
    for file in listdir(file_path):
        if file[-5:]=='.json':
            file_name = file[:-5]
    if file_name == 0:
        print('no labeling data')
    else:
        im = plt.imread(file_path+file_name+'.tif')
        try: h,w,N = np.shape(im)
        except: h,w = np.shape(im)
        shapes = json.load(open(file_path+file_name+'.json', 'r'))['shapes']
        n = len(shapes)
        masks = []
        for i in range(n):
            if shapes[i]['label']==tissue:
                points = np.asarray(shapes[i]['points'])
                Np,_ = np.shape(points)
                polygon = np.int_(np.reshape(points,Np*2))
                masks.append(create_mask(w,h,polygon))
        if masks == []:
            mask = np.zeros((h,w))
        else:
            masks = np.asarray(masks)
            Nm,_,_ = np.shape(masks)
            mask = np.sum(masks,axis=0)
        t2 = time.time()
        print("Duration :",t2-t1)
        return mask

def sel_polygon(npy_file,img):
    """
    this function creates a mask form the polygon selected manually

    Parameters
    ----------
    npy_file : TYPE : string
        DESCRIPTION : numpy file name of the polygon coordinates
    img : TYPE : array
        DESCRIPTION : HS image

    Returns
    -------
    mask : TYPE : matrix
        DESCRIPTION : mask

    """
    print("--Creating mask--")
    h,w,n = img.shape
    t1 = time.time()
    points = npy_file
    Np,_ = np.shape(points)
    polygon = np.int_(np.reshape(points,Np*2))
    masks = []
    masks.append(create_mask(w,h,polygon))
    if masks == []:
        mask = np.zeros((h,w))
    else:
        masks = np.asarray(masks)
        Nm,_,_ = np.shape(masks)
        mask = np.sum(masks,axis=0)
    t2 = time.time()
    print("Duration :",t2-t1)
    
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('polygon selected manually')
    plt.show()
    return mask

def sat_mask(data):
    """
    this function creates a saturation mask removing all the saturated pixels

    Parameters
    ----------
    data : TYPE : matrix
        DESCRIPTION : HS image divided by its maximum

    Returns
    -------
    mask : TYPE : matrix
        DESCRIPTION : mask

    """
    ind_sat = np.argwhere(data==1)
    h,w,n = data.shape
    N,_ = ind_sat.shape
    mask = np.ones((h,w))
    for k in range(N):
        i = ind_sat[k][0]
        j = ind_sat[k][1]
        mask[i,j]=0
        # print(i,j,mask_sat[i,j])
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('saturation mask')
    plt.show()
    return mask

def apply_mask(img,mask):
    print("--Applying mask--")
    t1 = time.time()
    h,w,n = np.shape(img)
    img_mask = np.zeros((h,w,n))
    for k in range(n):
        img_mask[:,:,k] = np.where(mask==1,img[:,:,k],mask*np.nan)
    t2 = time.time()
    print("Duration : ",t2-t1)
    return img_mask

def blocs(IMG,s):
    print("--Applying blocs--")
    t1 = time.time()
    H,W,n = np.shape(IMG)
    h,w = H//s,W//s
    rshp = IMG[:H-H%s,:W-W%s,:]
    rshp = np.reshape(rshp,(h,s,w,s,n))
    t2 = time.time()
    print("Duration :",t2-t1)
    return np.mean(rshp,axis=(1,3))

def map_SO2(IMG):
    count = np.count_nonzero(np.isnan(IMG[:,:,0]) == False)
    print("Number of non nan blocs :", count)
    print("--Generating SO2 map--")
    t1 = time.time()
    h,w,n = np.shape(IMG)
    SO2_map = np.ones((h,w))*np.nan
    Ct_map = np.ones((h,w))*np.nan
    LBD = np.linspace(440,720,29)
    # ref = open_camera_response(LBD)
    ref = open_camera_response(LBD,'cam2')
    for i in range(h):
        for j in range(w):
            if not np.isnan(IMG[i,j,0]):
                I = np.asarray(IMG[i,j,:])/65536
                # I = np.asarray(IMG[i,j,:])/255
                I = I/ref
                popt,I_fit = fit_I(LBD,I)
                SO2_map[i,j] = popt[2]
                Ct_map[i,j] = popt[1]
    t2 = time.time()
    print("Duration :", t2-t1)
    return SO2_map, Ct_map
