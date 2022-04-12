import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import utils

dates = ['20200929', '20201001', '20201003', '20201005', '20201007',
         '20201009', '20201011', '20201013', '20201015', '20201017',
         '20201019', '20201021', '20201023', '20201025', '20201027']

opt = utils.Config()

svm_layouts = {
        'lSVM1': 0.01,
        'lSVM2': 1,
        'lSVM3': 100,
        'kSVM1': [0.01, 0.001],
        'kSVM2': [0.01, 'scale'],
        'kSVM3': [0.01, 'auto'],
        'kSVM4': [1, 0.001],
        'kSVM5': [1, 'scale'],
        'kSVM6': [1, 'auto'],
        'kSVM7': [100, 0.001],
        'kSVM8': [100, 'scale'],
        'kSVM9': [100, 'auto']
    }

# for svm_layout_id, params_list in tqdm(svm_layouts.items()):
#     print(svm_layout_id, params_list)

utils.save_dict(svm_layouts, opt.mushroom_class)