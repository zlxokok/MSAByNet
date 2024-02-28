# import numpy as np
# import os
# import glob
# import pandas as pd
#
#
# img_list = []
# label_list = []
# path_img_list = []
# path_label_list = []
# a = '/mnt/ai2022/zlx/dataset/xun_yan_ce/2D_da/train'
# b = glob.glob('/mnt/ai2022/zlx/dataset/xun_yan_ce/2D_da/val')
# for i in b:
#     if i.split('/')[-1] == 'image':
#         img_list.append(glob.glob(i + '/*')[0])
#         b.remove(i)
# for i in b:
#     if i.split('/')[-1] == 'label':
#         label_list.append(glob.glob(i + '/*')[0])
#         b.remove(i)
# for i in b:
#     path_img_list.append(os.path.join(i, 'image'))
#     path_label_list.append(os.path.join(i, 'label'))
# for i in path_img_list:
#     if len(os.listdir(i)) != 0:
#         b = os.listdir(i)
#         img_list.append(os.path.join(i, b[0]))
#     else:
#         print(i)
# for i in path_label_list:
#     if len(os.listdir(i)) != 0:
#         b = os.listdir(i)
#         label_list.append(os.path.join(i, b[0]))
#     else:
#         print(i)
# df = pd.DataFrame()
# df['image'] = img_list
# df['label'] = label_list
# df.to_csv('//mnt/ai2022/zlx/dataset/xun_yan_ce/train_all.csv', index=False)
# print('a')

import os
import csv


folder_path = "/mnt/ai2022/zlx/dataset/QaTa_COV19/Val Set/img"
csv_path = "/mnt/ai2022/zlx/dataset/QaTa_COV19/Val Set/val.csv"


file_names = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        file_names.append(file_name)


with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image_name"])
    for file_name in file_names:
        writer.writerow([file_name])
