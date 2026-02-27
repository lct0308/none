import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import time
import os
import sys
sys.path.append("/data1/pbw/Project")
# from YOLOP_LaneATT.lib.models.laneatt import LaneATT
from YOLOP_LaneATT_RDI.lib.models.laneatt import LaneATT
from YOLOP_main.lib.models.YOLOP import MCnet,YOLOP
from YOLOP_main.lib.core.general import non_max_suppression,scale_coords,box_iou,ap_per_class
from YOLOP_main.lib.utils import plot_one_box
import concurrent.futures
import ctypes
import json
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

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

class YolopTRT(object):
    """
    description: Warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding: ', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        self.input_h = 384
        self.input_w = 640
        self.img_h = 360
        self.img_w = 640

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1

        detout = host_outputs[0]
        # segout = host_outputs[1].reshape( (self.batch_size, self.img_h,self.img_w))
        # laneout = host_outputs[2].reshape( (self.batch_size, self.img_h,self.img_w))

        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                detout[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )

            # Draw rectangles and labels on the original image
            img = batch_image_raw[i]
            nh = img.shape[0]
            nw = img.shape[1]
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                label="{}:{:.2f}".format( categories[int(result_classid[j])], result_scores[j])
                plot_one_box( box, img, label=label)

            # seg  = cv2.resize(segout[i], (nw, nh), interpolation=cv2.INTER_NEAREST)
            # lane = cv2.resize(laneout[i], (nw, nh), interpolation=cv2.INTER_NEAREST)
            # color_area = np.zeros_like(img)
            # color_area[seg==1]  = (0,255,0)
            # color_area[lane==1] = (0,0,255)
            # color_mask = np.mean(color_area, 2)
            # img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_area[color_mask != 0] * 0.5
            img = img.astype(np.uint8)

        return batch_image_raw, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (114, 114, 114)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        image = (image - (0.485, 0.456, 0.406)) /(0.229, 0.224, 0.225)
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

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

det_idx_range = [str(i) for i in range(0,25)]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MCnet(YOLOP)
model_dict = model.state_dict()
checkpoint = torch.load('/data1/pbw/Project/YOLOP_main/runs/MyDataset/_2023-07-28-03-26/epoch-149.pth', map_location=device)
checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
model_dict.update(checkpoint_dict)
model.load_state_dict(model_dict)
# model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# PLUGIN_LIBRARY = "/data1/pbw/Project/build/libmyplugins.so"
# engine_file_path = "/data1/pbw/Project/build/yolop.trt"
# ctypes.CDLL(PLUGIN_LIBRARY)
# categories = ["arrow"]
# yolop_wrapper = YolopTRT(engine_file_path)

# model_lane = LaneATT(72, 1280, 720, '/data1/pbw/Project/YOLOP_LaneATT/data/culane_anchors_freq.pt', 1000, 64)
# checkpoint = torch.load('/data1/pbw/Project/YOLOP_LaneATT/experiments/laneatt_yolop_mydataset/models/model_0054.pt', map_location=device)
# model_lane.load_state_dict(checkpoint["model"])
# model_lane = model_lane.to(device)
# model_lane.eval()

model_lane = LaneATT(72, 1280, 720, '/data1/pbw/Project/YOLOP_LaneATT_RDI/data/culane_anchors_freq.pt', 1000, 64)
checkpoint = torch.load('/data1/pbw/Project/YOLOP_LaneATT_RDI/experiments/laneatt_yolop_mydataset_rdi/models/model_0096.pt', map_location=device)
model_lane.load_state_dict(checkpoint["model"])
model_lane = model_lane.to(device)
model_lane.eval()

T_total = AverageMeter()
T_inf = AverageMeter()
T_draw = AverageMeter()
T_data = AverageMeter()
p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
# t = time_synchronized()
read_path = "/data1/pbw/Mydataset/images/2023-04-28-16-24-48/"
save_ori_path = "/data1/pbw/Project/visualization"
predictions_dir = "/data1/pbw/Project/save_results/lane_detection"
txt_path = "/data1/pbw/Project/save_results/arrow_detection"
for i in range(3):
    to_tensor = ToTensor()
    images = np.random.randint(low=0, high=255, size=(720, 1280, 3)).astype(np.uint8)
    rdi = np.random.randint(low=0, high=255, size=(23, 40)).astype(np.uint8)
    ori_image = images
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
    img = np.expand_dims(img, axis=0)  # 3维转4维
    img = torch.tensor(img)
    img = img.to(device)

    images = images.transpose(2, 0, 1)
    # img = img.astype(np.float32) / 255
    images = images.astype(np.float32)
    images = np.expand_dims(images, axis=0)  # 3维转4维
    images = torch.tensor(images)
    images = images.to(device)
    rdi = to_tensor(rdi)
    rdi = rdi.unsqueeze(0)
    rdi = rdi.to(device)

    output = model_lane(images, rdi)
    prediction = model_lane.decode(output, as_lanes=True)
    det_out = model(img)
    inf_out,train_out = det_out
    boxes = non_max_suppression(inf_out)[0]

input("press enter to start!")

# def collate_fn(batch):
#     orin, img, images, img_name = zip(*batch)
#     return np.array(orin), img, images, img_name
from dataset import MyDataset 
dataset = MyDataset()
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
plt.ion()
iouv = torch.linspace(0.5,0.95,10).to(device)     #iou vector for mAP@0.5:0.95
niou = iouv.numel()
jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
record_num = 0
with torch.no_grad():
    for orin, img, images, img_name, para, rdi, target in data_loader:
        img_name = img_name[0]
        img_base_name = img_name.split('/')[1]

        # visualization results save path
        save_path = os.path.join(save_ori_path, img_name)
        save_path = os.path.join(save_ori_path, img_name.split('/')[0])
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_ori_path, img_name)

        # inference results save path (lane detection + arrow detection)
        pred_path = os.path.join(predictions_dir, img_name)
        pred_path = os.path.join(predictions_dir, img_name.split('/')[0])
        os.makedirs(pred_path, exist_ok=True)
        pred_path = os.path.join(predictions_dir, img_name.replace('png','json'))

        txt_save_path = os.path.join(txt_path, img_name)
        txt_save_path = os.path.join(txt_path, img_name.split('/')[0])
        os.makedirs(txt_save_path, exist_ok=True)
        txt_save_path = os.path.join(txt_path, img_name.replace('png','txt'))

        # orin_path = os.path.join("/data1/pbw/Project/gt", img_name)
        # ori_image = cv2.imread(orin_path)

        total_time = time_synchronized()
        ori_image = orin.numpy().squeeze()
        # print(ori_image.shape)
        # cv2.imshow("detection_results", ori_image)
        r, dw, dh, new_unpad_w, new_unpad_h = para
        r = r.to(device)
        dw = dw.to(device)
        dh = dh.to(device)
        new_unpad_w = new_unpad_w.to(device)
        new_unpad_h = new_unpad_h.to(device)
        
        img = img.to(device)
        images = images.to(device)
        rdi = rdi.to(device)
        target = target.to(device)

        t = time_synchronized()
        output = model_lane(images, rdi)
        prediction = model_lane.decode(output, as_lanes=True)
        det_out = model(img)
        inf_out,train_out = det_out
        boxes = non_max_suppression(inf_out)[0]
        time1 = time_synchronized() - t
        print("cost time: ", time1)
        T_inf.update(time1,1)

        t = time_synchronized()
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r
        # images = images[:, :, ::-1].copy()
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            ori_image = cv2.rectangle(ori_image, (x1, y1), (x2, y2), (255, 0, 0), 2, 2)

            line = (label, x1, y1, x2, y2, conf)
            with open(txt_save_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
        predn = boxes
        labels = target[0]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        if len(predn) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
        else:
            correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                tbox = labels[:, 1:5]
                # Per target class
                for cls in torch.unique(tcls_tensor):                    
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == predn[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # n*m  n:pred  m:label
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))

        ys = np.arange(0,720,5) / 720   
        lane_lines = []
        lane_save = []
        result = {}
        prediction = prediction[0]
        for lane in prediction:
            # lane1 = lanes[0]
            # for lane in lanes:
            uv_orgsize = []
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs[valid_mask]
            lane_xs = xs * 1280
            lane_ys = ys[valid_mask] * 720
            lane_points = [(int(lane_xs[i]), int(lane_ys[i])) for i in range(len(lane_xs) - 1)]
            lane_lines.append(lane_points)
            uv_orgsize.append(list(lane_xs))
            uv_orgsize.append(list(lane_ys))
            lane_save.append({'uv': uv_orgsize})
        result['lane_lines'] = lane_save
        
        with open(pred_path, 'w') as result_file:
            json.dump(result, result_file)

        for lane_points in lane_lines:
            for i in range(len(lane_points) - 1):
                ori_image = cv2.line(ori_image, lane_points[i], lane_points[i+1], thickness=5, color=(255, 0, 0))
        
        # t2 = time_synchronized()

        # record_num = record_num + 1
        # if record_num % 10 == 0:
        #     cv2.namedWindow("detection_results", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("detection_results", 1280, 720)
        #     cv2.imshow("detection_results", ori_image)
        #     key = cv2.waitKey(1)
        
        # ori_image = Image.fromarray(ori_image)
        # plt.clf()
        # plt.imshow(ori_image)
        # plt.pause(0.001)
        # plt.ioff()
        # time2 = time_synchronized() - t2
        # T_data.update(time2,1)

        # Save the final image
        cv2.namedWindow('lane det', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('lane det', 640, 360)
        cv2.imshow('lane det', ori_image)
        cv2.waitKey(1)
        # cv2.imwrite(save_path, ori_image)
        time1 = time_synchronized() - t
        T_draw.update(time1,1)
        # print(f"lane draw time: {time1} s")

        total_time1 = time_synchronized() - total_time
        T_total.update(total_time1,1)

stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip
map70 = None
map75 = None
if len(stats) and stats[0].any():
    p, r, ap, f1, ap_class, n_l, n_p = ap_per_class(*stats, plot=False)
    ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
    mp, mr, map50, map70, map75, map, f1 = \
    p.mean(), r.mean(), ap50.mean(), ap70.mean(),ap75.mean(),ap.mean(),f1.mean()
tp = int(n_p * mp)
fn = int(tp / mr - tp)
msg = "GT num: {nl}, Pred num: {np}, TP: {tp}, FN:{fn}, Precision: {p:.6f}, Recall: {r:.6f}, F1: {F1:.6f}"\
    .format(nl=n_l, np=n_p, tp=tp, fn=fn, p=mp,r=mr,F1=f1)

print(f"average cost time: {T_inf.avg}")
metrics = MyDataset.eval_predictions(output_basedir=predictions_dir)
metrics_path = os.path.join(predictions_dir, 'test_metrics.json')
with open(metrics_path, 'w') as results_file:
    json.dump(metrics, results_file)
print("Lane Detection Results:\n", str(metrics))
print("Arrow Detection Results:\n", msg)

# for img_name in sorted(os.listdir(read_path)):
#     total_time = time_synchronized()
#     img_path = os.path.join(read_path, img_name)
#     save_path = os.path.join(save_ori_path, img_name)
#     images = cv2.imread(img_path)
#     # print(images.shape)
#     ori_image = images
#     img = images

#     t = time_synchronized()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img, (640, 640))

#     img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
#     img /= 255.0
#     img[:, :, 0] -= 0.485
#     img[:, :, 1] -= 0.456
#     img[:, :, 2] -= 0.406
#     img[:, :, 0] /= 0.229
#     img[:, :, 1] /= 0.224
#     img[:, :, 2] /= 0.225
    
#     img = img.transpose(2, 0, 1)
#     img = np.expand_dims(img, axis=0)  # 3维转4维
#     img = torch.tensor(img)
#     img = img.to(device)

#     images = images.transpose(2, 0, 1)
#     # img = img.astype(np.float32) / 255
#     images = images.astype(np.float32)
#     images = np.expand_dims(images, axis=0)  # 3维转4维
#     images = torch.tensor(images)
#     images = images.to(device)
#     time1 = time_synchronized() - t
#     T_data.update(time1,1)

#     t = time_synchronized()
#     output = model_lane(images)
#     prediction = model_lane.decode(output, as_lanes=True)
#     det_out = model(img)
#     inf_out,train_out = det_out
#     boxes = non_max_suppression(inf_out)[0]
#     time1 = time_synchronized() - t

#     # # trt
#     # batch_image_raw, use_time = yolop_wrapper.infer([ori_image])
#     # ori_image = batch_image_raw
#     # time1 = time_synchronized() - t

#     T_inf.update(time1,1)
#     # print(f"one flame time: {time1} s")

#     t = time_synchronized()
#     boxes[:, 0] -= dw
#     boxes[:, 1] -= dh
#     boxes[:, 2] -= dw
#     boxes[:, 3] -= dh
#     boxes[:, :4] /= r
#     # images = images[:, :, ::-1].copy()
#     for i in range(boxes.shape[0]):
#         x1, y1, x2, y2, conf, label = boxes[i]
#         x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
#         ori_image = cv2.rectangle(ori_image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
#     # time1 = time_synchronized() - t
#     # T_arrow_draw.update(time1,1)
#     # print(f"arrow draw time: {time1} s")
    
#     # t = time_synchronized()
#     ys = np.arange(0,720,5) / 720   
#     lane_lines = []
#     result = {}
#     prediction = prediction[0]
#     for lane in prediction:
#         # lane1 = lanes[0]
#         # for lane in lanes:
#         uv_orgsize = []
#         xs = lane(ys)
#         valid_mask = (xs >= 0) & (xs < 1)
#         xs = xs[valid_mask]
#         lane_xs = xs * 1280
#         lane_ys = ys[valid_mask] * 720
#         lane_points = [(int(lane_xs[i]), int(lane_ys[i])) for i in range(len(lane_xs) - 1)]
#         lane_lines.append(lane_points) 
#     # def process_lane(lane):
#     #     uv_orgsize = []
#     #     xs = lane(ys)
#     #     valid_mask = (xs >= 0) & (xs < 1)
#     #     xs = xs[valid_mask]
#     #     lane_xs = xs * 1280
#     #     lane_ys = ys[valid_mask] * 720

#     #     # Collect lane points for later use
#     #     lane_points = [(int(lane_xs[i]), int(lane_ys[i])) for i in range(len(lane_xs) - 1)]
#     #     return lane_points

#     # # Parallel processing of lanes
#     # with concurrent.futures.ThreadPoolExecutor() as executor:
#     #     lane_points_list = list(executor.map(process_lane, prediction[0]))

#     for lane_points in lane_lines:
#         for i in range(len(lane_points) - 1):
#             ori_image = cv2.line(ori_image, lane_points[i], lane_points[i+1], thickness=2, color=(0, 0, 255))
#     cv2.namedWindow("detection_results", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("detection_results", 1280, 720)
#     cv2.imshow("detection_results", ori_image)
#     key = cv2.waitKey(1)
#     # Save the final image
#     # cv2.imwrite(save_path, ori_image)
#     time1 = time_synchronized() - t
#     T_draw.update(time1,1)
#     # print(f"lane draw time: {time1} s")

#     total_time1 = time_synchronized() - total_time
#     T_total.update(total_time1,1)
# print(f"total time: {T_total.avg}, average cost time: {T_inf.avg}, average draw time: {T_draw.avg}")
# , average imshow time: {T_data.avg}

