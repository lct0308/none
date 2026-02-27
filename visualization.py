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
file_list_path = "/data1/pbw/Project/test_seq/crowded.txt"
save_dir = "/data1/pbw/Project/gt_pred"

with open(file_list_path, 'r') as file_list:
    for line in file_list:
        line = line.split()[0]
        img_path = os.path.join('/data1/pbw/Project/visualization', line)   
        line = line.replace('png','json')
        gt_path = os.path.join(anno_dir, line)
        temp = line
        save_path = os.path.join(save_dir, temp.split('/',1)[0])
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_dir, line.replace('json','png'))

        img = cv2.imread(img_path)
        anno_lanes = load_gt_data(gt_path)

        image_save = np.zeros((720, 1280), dtype=np.uint8)
        # 绘制车道标注点
        for i in range(len(anno_lanes)):
            uv = anno_lanes[i]
            for j in range(len(uv)-1):
                image_save = cv2.line(img, (int(uv[j][0]), int(uv[j][1])), (int(uv[j+1][0]), int(uv[j+1][1])), thickness=2, color=(0, 0, 255))

        with open(gt_path, 'r') as f:
            gt = json.load(f)
            labels = gt.get('dataList')
        for label in labels:
            if label["label"] == "Arrowsymbol":
                coordinates = label["coordinates"]
                image_save = cv2.rectangle(image_save, (int(coordinates[0][0]), int(coordinates[0][1])), (int(coordinates[1][0]), int(coordinates[1][1])), thickness=2, color=(0, 0, 255))

        cv2.imwrite(save_path, image_save)