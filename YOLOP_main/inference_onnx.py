import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import time
import os
import concurrent.futures
from lib.models.YOLOP import MCnet,YOLOP
from lib.core.general import non_max_suppression,scale_coords
from lib.utils import plot_one_box

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

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MCnet(YOLOP)
checkpoint = torch.load('/data1/pbw/YOLOP_main/runs/MyDataset/_2023-07-28-03-26/epoch-149.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

T_total = AverageMeter()
T_inf = AverageMeter()
T_arrow_draw = AverageMeter()
# t = time_synchronized()
read_path = "/data1/pbw/Mydataset/images/2023-04-28-16-24-48/"
save_ori_path = "/data1/pbw/YOLOP_main/images_results_onnx"
ONNX_Model_Path = '/data1/pbw/YOLOP_main/weights/arrow.onnx'
for i in range(3):
    for img_name in sorted(os.listdir(read_path)):
        total_time = time_synchronized()
        img_path = os.path.join(read_path, img_name)
        save_path = os.path.join(save_ori_path, img_name)
        ori_image = cv2.imread(img_path)
        img = ori_image
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
        img = np.expand_dims(img, axis=0)  # 3维转4维
        # img = torch.tensor(img)
        # img = img.to(device)

        t = time_synchronized()
        # det_out, _, _= model(img)
        # inf_out,train_out = det_out
        # boxes = non_max_suppression(inf_out)[0]
        sess = ort.InferenceSession(ONNX_Model_Path)
        # 模型的输入输出名，必须和onnx的输入输出名相同，可以通过netron查看，如何查看参考下文
        input_name = "images"
        output_name=['det_out', 'drive_area_seg', 'lane_line_seg']
        # run方法用于模型推理，run(输出张量名列表，输入值字典)
        t = time_synchronized()
        det_out, _, _ = sess.run(output_name, {input_name: img})
        det_out = torch.tensor(det_out)
        boxes = non_max_suppression(torch.tensor(det_out), conf_thres= 0.2, iou_thres=0.5)[0]
        # output = non_max_suppression(det_out, conf_thres= 0.0001, iou_thres=0.1)
        time1 = time_synchronized() - t
        if i >= 2:
            T_inf.update(time1,1)
        # print(f"one flame time: {time1} s")

        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r
        # images = images[:, :, ::-1].copy()
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            ori_image = cv2.rectangle(ori_image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        
        cv2.imwrite(save_path, ori_image)

        total_time1 = time_synchronized() - total_time
        if i >= 2:
            T_total.update(total_time1,1)
print(f"total time: {T_total.avg}, average cost time: {T_inf.avg}")
# time1 = time_synchronized() - t
# print(f"total time: {time1} s")
# for lane in prediction:
#     lane1 = list(lane[0])
#     print(lane1)
#     for j in range(len(lane1)-1):
#         print(lane1[j][0], lane1[j][1])
#         images = cv2.line(images, (int(lane1[j][0] * 1280), int(lane1[j][1] * 720)), (int(lane1[j+1][0] * 1280), int(lane1[j+1][1] * 720)), thickness=2, color=(0, 0, 255))
#     # for point in lane:
#     #     x, y = point
#     #     x, y = int(x * 1280), int(y * 720)
#     #     image_save = cv2.line(img, (int(lane[j][0]), int(lane[j][1])), (int(lane[j+1][0]), int(lane[j+1][1])), thickness=2, color=(0, 0, 255))

# # 保存可视化结果
