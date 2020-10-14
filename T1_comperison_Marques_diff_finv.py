# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:56:39 2020

@author: musti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




f_inv = [0.96,0.92,0.88,0.84,0.80]
T1_Marques = [1211.801, 1272.432 , 1323.174, 1600, 1700]

stdDiv = [79.2 , 88.598, 90.795, 90, 90]


for i in range(len(f_inv)):
    plt.errorbar(f_inv[i], T1_Marques[i], yerr=stdDiv[i],marker='s', label='{}'.format(f_inv[i]))
plt.grid(color='k', linestyle=':', linewidth=1)
plt.title('Inversion Effeciency vs T1-values')
plt.xlabel('Inversion Effeciency f_inv')
plt.ylabel('T1-values')