import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import random
# from nms import non_max_suppression, scale_coords, xyxy2xywh
from lib.core.general import non_max_suppression,check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,coco80_to_coco91_class,plot_images,ap_per_class,output_to_target
from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask,plot_one_box,show_seg_result

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

input_image_path = "/data1/pbw/Mydataset/images/2023-04-28-16-24-48/1682670314.649735.png"
ONNX_Model_Path = '/data1/pbw/YOLOP_main/weights/arrow.onnx'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_inf = AverageMeter()
def onnx_infer(im, onnx_model):
 
    # InferenceSession获取onnxruntime解释器
    sess = ort.InferenceSession(onnx_model)
    # 模型的输入输出名，必须和onnx的输入输出名相同，可以通过netron查看，如何查看参考下文
    input_name = "images"
    output_name=['det_out', 'drive_area_seg', 'lane_line_seg']
    # run方法用于模型推理，run(输出张量名列表，输入值字典)
    t = time_synchronized()
    det_out, da_seg_out, ll_seg_out = sess.run(output_name, {input_name: im})
    det_out = torch.tensor(det_out)
    # print("output success")
    # print(det_out)
    # output = non_max_suppression(torch.tensor(det_out), conf_thres= 0.2, iou_thres=0.5)
    output = non_max_suppression(det_out, conf_thres= 0.0001, iou_thres=0.1)
    # det_out = np.array(det_out)
    # print(output)
    time1 = time_synchronized() - t
    T_inf.update(time1,1)
    print(f"one flame time: {time1} s")

    img_det = cv2.imread(input_image_path)
    # print(img_det.shape)
    # det = torch.tensor(det_out)
    # pred0 = torch.tensor(det_out)
    det = output[0].clone()
    pred0 = output[0].clone()
    if len(det):
        det[:,:4] = scale_coords([384, 640],det[:,:4], img_det.shape).round()
        pred0[:, :4] = scale_coords([384, 640], pred0[:, :4], img_det.shape).round()
        print(max(det[:,4]))
    # print(det)
    for *xyxy,conf,cls in (det.squeeze()):
        # print(xyxy)
        # xywh = (xyxy2xywh(np.array(xyxy))).tolist()
        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        # if xywh[2] < 25 or xywh[3] < 25:
        #     continue
        plot_one_box(xyxy, img_det, color=[255,0,0], line_thickness=3)
    cv2.imwrite("1.png",img_det)
    print(f"average cost time: {T_inf.avg}")

 
if __name__ == '__main__':
    # 图片的预处理
    img = cv2.imread(input_image_path)
    img = cv2.resize(img, (640, 640))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)  # 3维转4维
 
    output = onnx_infer(img, ONNX_Model_Path)