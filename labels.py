# -*- coding: utf-8 -*-
"""
@author: victoire
Imperial College London - MSc 2022 - Biomedical engineering
Individual project - Hyperspectral images analysis for neurosurgery

labels.py
in this file
functions to plot the labels of the tissue segmented by the specialist
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from os import listdir

# %%

def label2color(label):
    if label == 'Tumour Core':
        return 'k'
    if label == 'Tumour Margins':
        return 'gray'
    if label == 'Dura':
        return 'y'
    if label == 'Blood vessel':
        return 'r'
    if label == 'Arachnoid':
        return 'g'
    if label == 'Blood':
        return 'orange'
    if label == 'Cortical Surface':
        return 'c'
    if label == 'Bone':
        return 'w'

def draw_region(ax,shapes,i):
    points = np.asarray(shapes[i]['points']).T
    label = shapes[i]['label']
    # if label == 'Dura':
    color = label2color(label)
    ax.plot(points[0],points[1],label=label,color=color)

def plot_img_labels(file_path):
    file_name = 0
    for file in listdir(file_path):
        if file[-5:]=='.json':
            file_name = file[:-5]
    if file_name == 0:
        print('no labeling data')
    else:
        fig,ax = plt.subplots(figsize = (15,8))
        file_name = file_path + file_name
        try:
            im = plt.imread(file_name+'.tif')
            ax.imshow(im, cmap ='gray')
        except:
            im = plt.imread(file_name+'.jpg')
            ax.imshow(im)
        shapes = json.load(open(file_name+'.json', 'r'))['shapes']
        n = len(shapes)
        
        for i in range(n):
            draw_region(ax,shapes,i)
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()  
        lgd = dict(zip(labels, handles))
        ax.legend(lgd.values(), lgd.keys())
        plt.title(file_name)
        plt.tight_layout()
    plt.show()
