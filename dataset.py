from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import json
import glob
import cv2
import numpy as np
from tqdm import tqdm
import YOLOP_LaneATT_RDI.utils.openlane_metric as culane_metric

id_dict_single = {'Arrowsymbol': 0}

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)

test_txt = "/data1/pbw/Project/test_seq/demo.txt" #crowded/final_test

class MyDataset(Dataset):
    def __init__(self, directory = "/data1/pbw/Mydataset/images/", transform=None):
        self.directory = directory
        self.rdi_path = "/data1/pbw/Mydataset/rdi/"
        self.transform = transform
        self.image_filenames = []
        with open(test_txt, 'r') as list_file:
            for file in list_file:
                file = file.split()
                img_line = file[0]
                self.image_filenames.append(img_line)
        
        # from random import sample
        # random_num = 1102
        # self.image_filenames = sample(self.image_filenames, random_num)

        self.to_tensor = ToTensor()
        self.gt_path = "/data1/pbw/Mydataset/label/"
        # self.image_filenames = [f for f in os.listdir(directory) if f.endswith('.png')]
        # self.gt_path = "/data1/pbw/RDI_net/val/RDI"
        # self.image_filenames = glob.glob(r"/data1/pbw/Mydataset/images/*/*.png")
        # for i in range(len(self.image_filenames)):
        #     self.image_filenames[i] = self.image_filenames[i].split('/', 5)[5]
        # self.image_filenames = sorted(self.image_filenames)
    
    def filter_data(self, data):
        remain = []
        for obj in data:
            if obj["label"] == "Arrowsymbol":  # obj.has_key('box2d'):
                cha = []
                cha.append(obj["coordinates"][1][0] - obj["coordinates"][0][0])
                cha.append(obj["coordinates"][1][1] - obj["coordinates"][0][1])
                if cha[0] < 25 or cha[1] < 25:
                    continue
                if obj['label'] in id_dict_single.keys():
                    remain.append(obj)
        return remain

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        rdi_path = os.path.join(self.rdi_path, img_name)
        img_path = os.path.join(self.directory, img_name)
        label_path = os.path.join(self.gt_path, img_name.replace('png','json'))

        # import shutil
        # new_path =  "/data1/pbw/Project/test_images"
        # new_path = os.path.join(new_path, img_name.split('/')[0])
        # os.makedirs(new_path, exist_ok=True)
        # shutil.copy(img_path, new_path)

        # import shutil
        # new_path =  "/data1/pbw/Project/rdi_images"
        # new_path = os.path.join(new_path, img_name.split('/')[0])
        # os.makedirs(new_path, exist_ok=True)
        # shutil.copy(rdi_path, new_path)

        # load arrow label
        with open(label_path, 'r') as f:
            label = json.load(f)
            data = label.get('dataList')
        data = self.filter_data(data)
        gt = np.zeros((len(data), 5))
        for idx, obj in enumerate(data):
            x1 = float(obj['coordinates'][0][0])
            y1 = float(obj['coordinates'][0][1])
            x2 = float(obj['coordinates'][1][0])
            y2 = float(obj['coordinates'][1][1])
            cls_id=0
            gt[idx][0] = cls_id
            gt[idx][1:] = list((x1, y1, x2, y2)) # 目标检测的gt第一项是类别，后四项为框中心坐标和宽高的归一化
        
        rdi = cv2.imread(rdi_path, 0)
        rdi = cv2.resize(rdi, (40, 23), interpolation=cv2.INTER_AREA)
        # rdi = np.expand_dims(rdi, axis=0)
        rdi = self.to_tensor(rdi)

        images = cv2.imread(img_path)
        orin = images
        # print(images.shape)
        img = images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img, (640, 640))

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225
        
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img)

        images = images.transpose(2, 0, 1)
        # img = img.astype(np.float32) / 255
        images = images.astype(np.float32)
        images = torch.tensor(images)

        # return image, label, img_name
        return orin, img, images, img_name, (r, dw, dh, new_unpad_w, new_unpad_h), rdi, gt
    
    def eval_predictions(output_basedir):
        print('Generating prediction output...')
        anno_root = "/data1/pbw/Mydataset/label/"
        list = test_txt
        official_metric = True
        return culane_metric.eval_predictions(output_basedir, anno_root, list, official=official_metric)
