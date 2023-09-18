# -*- coding: utf-8 -*-
"""
@author: victoire
Imperial College London - MSc 2022 - Biomedical engineering
Individual project - Hyperspectral images analysis for neurosurgery

main.py
in this code
1) reading HS image
2) fitting algorithm for 1 pixel
3) draw region of interest for blood oxygenation level mapping
4) generates map and stores it in results folder

Run each bloc one after the other

"""

# %% libraries

import time
import os
import matplotlib.pyplot as plt
import numpy as np
from functions import blocs, map_SO2
from functions import apply_mask
from functions import openimg
from functions import filename_func, draw_polygon
from functions import ask_click
from functions import sel_polygon, sat_mask, seg_polygon
from display import plot_img_labels

# %% BLOC 1

# HS image path
folder1 = '../../../../Individual project/'
folder2 = 'Registered data/'
folder3 = 'HS029/'
field = 'Field01/'
Run = 'Run03/'
file_path = folder1 + folder2 + folder3 + field
img_path = file_path+Run
title = folder2+folder3+Run+", at 620nm" # title for plots

# creating a folder for the results (with date and time as a folder name)
foldername = filename_func(time.localtime())
results_path = 'SO2_results/'+foldername
os.mkdir(results_path)
with open(results_path+'/info.txt', 'w') as f:
    f.write(img_path)

# opening image
img = openimg(img_path)

# plot segmentation by spetialist
plot_img_labels(file_path)

# %% BLOC 2

# ask the user to click on one pixel to plot fit
ask_click(img,title)


# %% BLOC 3

# ask the user to draw a ploygon to generate map
draw_polygon(img,results_path)


# %% BLOC 4

# creating a mask to remove saturated pixels
mask1 = sat_mask(img/img.max())

# creating another mask according to the region of interest (uncomment desired line)
# --> either the polygon created previously
mask2 = sel_polygon(np.load(results_path+'/polygon.npy'),img)
# --> or one tissue segmented by the specialist
# mask2 = seg_polygon(file_path,'Dura')

# creating a mask combination of mask 1 and mask 2
mask = ((mask1+mask2)==2)

plt.figure()
plt.imshow(mask, cmap = 'gray')
plt.title('mask combination')
plt.show()

# applying the mask on the image
img2 = apply_mask(img, mask)

# reducing the resolution of the HS image by averaging the pixels by blocs of 20 pixels
img3 = blocs(img2, 50)

# creating SO2 map and Ct map
SO2_map, Ct_map = map_SO2(img3)

# saving in results folder
np.save(results_path+'/SO2_50.npy', SO2_map)
np.save(results_path+'/Ct_50.npy', Ct_map)



