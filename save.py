import cv2
import numpy as np
import json
import os
import time
import torch

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def load_pred_data(pred_path):
    with open(pred_path, 'r') as file:
        file_lines = [line for line in file]
        info_dict = json.loads(file_lines[0])
        lanes = info_dict['lane_lines']
        img = []
        for idx in range(len(lanes)):
            lane1 = lanes[idx]
            uv1 = np.array(lane1['uv'])
            img_data = [[uv1[0][i], uv1[1][i]] for i in range(0, uv1.shape[1])]
            img.append(img_data)

    return img

def load_gt_data(gt_path):
    with open(gt_path, 'r') as file:
        result = json.load(file)
        dataList = result.get('dataList')
        lanes = dataList
        img = []
        for idx in range(len(lanes)):
            lane1 = lanes[idx]
            if lane1['shapeType'] != "line":
                continue
            uv1 = np.array(lane1['coordinates']).T
            img_data = [[uv1[0][i], uv1[1][i]] for i in range(0, uv1.shape[1])]
            img.append(img_data)

    return img

T_draw = AverageMeter()
anno_dir = "/data1/pbw/Mydataset/label/"
file_list_path = "/data1/pbw/Project/normal.txt"
save_dir = "/data1/pbw/Project/normal"

with open(file_list_path, 'r') as file_list:
    for line in file_list:
        line = line.split()[0]
        orin_path = os.path.join('/data1/pbw/Mydataset/images/', line)
        pred_path = os.path.join('/data1/pbw/Project/visualization', line)
        gt_path = os.path.join('/data1/pbw/Project/gt', line)
        gt_pred_path = os.path.join('/data1/pbw/Project/gt_pred', line)
        rdi_path = os.path.join('/data1/pbw/Project/rdi_images', line)

        temp = line
        save_path = os.path.join(save_dir, temp.split('/',1)[0])
        os.makedirs(save_path, exist_ok=True)
        
        import shutil
        shutil.copy(orin_path, os.path.join(save_path, 'original_image.png'))
        shutil.copy(pred_path, os.path.join(save_path, 'pred.png'))
        shutil.copy(gt_path, os.path.join(save_path, 'gt.png'))
        shutil.copy(gt_pred_path, os.path.join(save_path, 'gt_pred.png'))
        shutil.copy(rdi_path, os.path.join(save_path, 'rdi.png'))
