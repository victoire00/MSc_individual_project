# -*- coding: utf-8 -*-
"""
@author: victoire
Imperial College London - MSc 2022 - Biomedical engineering
Individual project - Hyperspectral images analysis for neurosurgery

montecarlo.py
in this file:
the code to run the monte carlo simulation
"""

from functions import save_diff_comp
from classes import parameters
import numpy as np

params = parameters()
LBD = np.linspace(400,780,101)
save_diff_comp(params,LBD)