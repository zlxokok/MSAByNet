import os
import torch
from tools_seg import Miou
import cv2
import argparse
from torch.utils.data import DataLoader
import numpy as np

import sys
import torchvision.transforms.functional as tf
from dataset.create_dataset_rgb import Mydataset, for_train_transform, test_transform
import pandas as pd
from dataset.create_dataset_rgb import Mydataset_test
from .MSAByNet import MSAByNet
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_val_path', '-iv', type=str,
                    default='', help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='', help='labels val data path.')
parser.add_argument('--resize', default=512, type=int, help='resize shape')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--imgs_val_list', '-cv', type=str,
                    default='/mnt/ai2022/zlx/dataset/Dataset_BUSI/test.csv', help='labels val data path.'),)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
result_path = '/mnt/ai2022/zlx/第二篇/jieguo/BUSI/MSAByNet'
if not os.path.exists(result_path):
    os.mkdir(result_path)
label_path = '/mnt/ai2022/zlx/CCM-SEG/base/result/label/'
val_csv = pd.read_csv(args.imgs_val_list)#[:30]
val_imgs, val_masks = val_csv['image_name'], val_csv['image_name']


val_imgs = [''.join(['/mnt/ai2022/zlx/dataset/Dataset_BUSI/image_test','/',i]) for i in val_imgs]
val_masks = [''.join(['/mnt/ai2022/zlx/dataset/Dataset_BUSI/mask_test','/',i]) for i in val_masks]
# val_imgs = [cv2.imread(i,cv2.IMREAD_UNCHANGED) for i in val_imgs]
# val_masks = [cv2.imread(i,cv2.IMREAD_UNCHANGED) for i in val_masks]

train_transform = for_train_transform()
test_transform = test_transform
valset = Mydataset_test(val_imgs, val_masks, test_transform)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
print('==> Preparing data..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
print('==> Building model..')

# model.encoder.load_state_dict(torch.load('tools_seg/resnet34-333f7ec4.pth'))
model = model.to('cuda')

state_dict = torch.load('/mnt/ai2022/zlx/第二篇/BUSI/MSAByNet/ckpt.pth')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

import csv
from PIL import Image
import time

begin_time = time.time()

imwrite_image = True


def train_val():
    with torch.no_grad():
        train_SP = 0
        PA = 0
        list = []
        number = 0
        train_dice = 0
        train_jaccard = 0
        train_accuracy = 0
        train_recall = 0
        train_precision = 0
        train_mdice = 0
        train_miou = 0
        train_pa = 0
        train_f1 = 0
        for batch_idx, (name, imgs, masks) in enumerate(valloader):
            number += 1
            # print(img_path)
            sys.stdout.write('\r%d/%s' % (number, len(valloader)))
            batch_idx += 1
            imgs, masks_cuda = imgs.to(device), masks.to(device)

            imgs = imgs.float()
            masks_pred = model(imgs)
            masks_pred = masks_pred[0]

            predicted = masks_pred.argmax(1)
            train_mdice += Miou.calculate_mdice(predicted, masks_cuda, 2).item()
            train_miou += Miou.calculate_miou(predicted, masks_cuda, 2).item()
            # train_pa += Miou.Pa(predicted, masks_cuda).item()
            # train_pre += Miou.pre(predicted, masks_cuda).item()
            # train_recall += Miou.recall(predicted, masks_cuda).item()
            # train_F1score += Miou.F1score(predicted, masks_cuda).item()
            train_jaccard += Miou.jaccard(predicted, masks_cuda).item()
            train_accuracy += Miou.accuracy(predicted, masks_cuda).item()
            train_dice += Miou.dice(predicted, masks_cuda).item()
            train_recall += Miou.recall(predicted, masks_cuda).item()
            train_SP += Miou.SP(predicted, masks_cuda).item()
            train_precision += Miou.precision(predicted, masks_cuda).item()
            # train_pa += Miou.Pa(predicted, masks_cuda).item()
            train_f1 += Miou.F1score(predicted, masks_cuda).item()


            #  softmax
            if imwrite_image:
                predict = predicted.squeeze(0)
                mask_np = predict.cpu().numpy()  # np.array
                mask_np = (mask_np * 255).astype('uint8')
                mask_np[mask_np > 0] = 255
                cv2.imwrite(os.path.join(result_path, name[0]), mask_np)
                masks_cuda = masks_cuda.squeeze(0)
                label_np = masks_cuda.cpu().numpy()  # np.array
                label_np = (label_np * 255).astype('uint8')
                label_np[label_np > 0] = 255
                masks_cuda_max = torch.max(masks_cuda)
                label_np_max = np.max(label_np)
                cv2.imwrite(os.path.join(label_path, name[0]), label_np)

        end_time = time.time()
        print('\n')
        print("时间")
        print(end_time - begin_time)
        # print('\n')
        # print(PA/number)
        print('jaccard')
        print(train_jaccard / number)
        print("accuracy")
        print(train_accuracy / number)
        print('dice')
        print(train_dice / number)
        print('recall')
        print(train_recall / number)
        print('SP')
        print(train_SP / number)
        print('precision')
        print(train_precision / number)
        print('meandice')
        print(train_mdice / number)
        print('miou')
        print(train_miou / number)
        # print('pa')
        # print(train_pa / number)
        print('f1')
        print(train_f1 / number)


# 40s 单张按批次  226.8个合一batch  145 4角度
if __name__ == '__main__':
    train_val()

#平均MIOU 0.9282328828178149
