import numpy as np
import time

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[np.arange(len(l)), l[:, 0].astype(int) + 5] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = np.where(x[:, 5:] > conf_thres)
            x = np.concatenate((box[i], x[i, j + 5][:, None], j[:, None].astype(float)), axis=1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            i = np.where(conf.view(-1) > conf_thres)[0]
            x = np.concatenate((box, conf, x[i, 5][:, None].astype(float)), axis=1)

        # Filter by class
        if classes is not None:
            class_mask = np.any(x[:, 5:6] == np.array(classes), axis=1)
            x = x[class_mask]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        if len(i) > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou.astype(float) * scores[np.newaxis]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]).astype(float) / weights.sum(axis=1, keepdims=True)  # merged boxes
            if redundant:
                i = i[np.sum(iou, axis=1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def nms(boxes, scores, iou_threshold):
    # Non-maximum suppression implementation
    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    scores = scores[sorted_idx]
    picked_indices = []

    while len(sorted_idx) > 0:
        picked_indices.append(sorted_idx[0])
        current_box = boxes[0]
        ious = box_iou(current_box, boxes[1:])
        overlapping_indices = np.where(ious > iou_threshold)[0]
        sorted_idx = np.delete(sorted_idx, overlapping_indices + 1)
        boxes = np.delete(boxes, overlapping_indices + 1, axis=0)
        scores = np.delete(scores, overlapping_indices + 1)

    return np.array(picked_indices)

def xywh2xyxy(x):
    y = x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y + h/2
    return y

def box_iou(box1, box2):
    # Calculate the Intersection over Union (IoU) of two boxes
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection

    return intersection / union

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 2] -= pad[0]
    coords[:, 1] -= pad[1]  # y padding
    coords[:, 3] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = abs(x[2] - x[0])  # width
    y[3] = abs(x[3] - x[1])  # height
    return y