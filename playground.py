import os
from tqdm import tqdm
from utils import Config

dates = ['20200929', '20201001', '20201003', '20201005', '20201007',
         '20201009', '20201011', '20201013', '20201015', '20201017',
         '20201019', '20201021', '20201023', '20201025', '20201027']

configs = Config()
# print("Importing data...")
# for folder_idx in tqdm(range(0, 30, 2)):
#     if configs.derivative:
#         img_folder = "A" + str(folder_idx) + '_DIFF/'
#     else:
#         img_folder = "A" + str(folder_idx) + '_NO_DIFF/'

#     cur_folder_path = 'D:/Repos/Python/freshness-decay/data/' + \
#         dates[folder_idx//2] + '/' + img_folder

#     print(cur_folder_path)

for i in range(0, 30, 2):
    print((i//2/14))
