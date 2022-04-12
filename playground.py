import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import utils

dates = ['20200929', '20201001', '20201003', '20201005', '20201007',
         '20201009', '20201011', '20201013', '20201015', '20201017',
         '20201019', '20201021', '20201023', '20201025', '20201027']

opt = utils.Config()

svm_derivative = {
        'lSVM1': 0.99,
        'lSVM2': 0.2,
        'lSVM3': 0.15,
        'kSVM1': 0.6,
        'kSVM2': 0.89,
        'kSVM3': 0.45,
        'kSVM4': 0.10,
        'kSVM5': 0.85,
        'kSVM6': 0.42,
        'kSVM7': 0.26,
        'kSVM8': 0.75,
        'kSVM9': 0.92
    }

svm_reflectance = {
        'lSVM1': 0.4,
        'lSVM2': 0.2,
        'lSVM3': 0.15,
        'kSVM1': 0.45,
        'kSVM2': 0.75,
        'kSVM3': 0.32,
        'kSVM4': 0.05,
        'kSVM5': 0.8,
        'kSVM6': 0.21,
        'kSVM7': 0.1,
        'kSVM8': 0.66,
        'kSVM9': 0.78
    }

x = ['lSVM1', 'lSVM2', 'lSVM3', 'kSVM1', 'kSVM2', 'kSVM3', 'kSVM4', 'kSVM5', 'kSVM6', 'kSVM7', 'kSVM8', 'kSVM9']
x_axis = np.arange(len(x))
svm_der = list(svm_derivative.values())
svm_ref = list(svm_reflectance.values())

plt.bar(x_axis - 0.2, svm_der, 0.4, label = 'Derivative Spectrum')
plt.bar(x_axis + 0.2, svm_ref, 0.4, label = 'Reflectance Spectrum')
  
plt.xticks(x_axis, x)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("")
plt.legend()
plt.show()