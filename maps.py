# -*- coding: utf-8 -*-
"""
@author: victoire
Imperial College London - MSc 2022 - Biomedical engineering
Individual project - Hyperspectral images analysis for neurosurgery

maps.py
in this file
1) code to read the results obtained by running main.py BLOC 4
2) code to plot the maps
"""

import numpy as np
import matplotlib.pyplot as plt
from display import plot_SO2_map, plot_Ct_map

# %% 1) read results

# write the path of your results :

# results_path = 'SO2_results/202391_11h12min16/'
# results_path = 'SO2_results/202391_12h44min9/'
# results_path = 'SO2_results/202391_13h17min54/'

results_path = 'SO2_results/202392_10h30min2/'
# results_path = 'SO2_results/202392_11h5min28/'
# results_path = 'SO2_results/202392_11h25min32/'
# results_path = 'SO2_results/202392_14h23min32/'

SO2_map_name = 'SO2_20.npy'
Ct_map_name = 'Ct_20.npy'

SO2_map = np.load(results_path+SO2_map_name)
Ct_map = np.load(results_path+Ct_map_name)

# %% 2) plot maps

plot_SO2_map(SO2_map)
plt.title('Oxygen saturation - Run04')
plt.tight_layout()
plot_Ct_map(Ct_map)
plt.title('Total hemoglobin concentration - Run 04')
plt.tight_layout()

# %% 3) compare 2 maps

results_path1 = 'SO2_results/202392_14h48min2/' # run 08
results_path2 = 'SO2_results/202392_15h38min54/' # run 06

SO2_map_name = 'SO2_10.npy'
Ct_map_name = 'Ct_10.npy'

SO2_map1 = np.load(results_path1+SO2_map_name)
Ct_map1 = np.load(results_path1+Ct_map_name)

SO2_map2 = np.load(results_path2+SO2_map_name)
Ct_map2 = np.load(results_path2+Ct_map_name)

# plot_SO2_map(SO2_map2)
# plt.title('Oxygen saturation - Run06')
# plt.tight_layout()
# plot_Ct_map(Ct_map2)
# plt.title('Total hemoglobin concentration - Run 06')
# plt.tight_layout()

diff1 = SO2_map1-SO2_map2
diff2 = Ct_map1-Ct_map2

# %%

plot_SO2_map(diff1)
plt.title('Oxygen saturation difference - Run03-Run05')
plot_Ct_map(diff2)
plt.title('Total hemoglobin concentration difference - Run03-Run05')
