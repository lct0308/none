from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from skimage import io

import cv2
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import os
import json

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def draw_lane(lane, img=None, img_shape=None, width=30):
  lane = np.array(lane)
  if img is None:
      img = np.zeros(img_shape, dtype=np.uint8)
  lane = lane.astype(np.int32)
  for p1, p2 in zip(lane[:-1], lane[1:]):
    cv2.line(img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
  return img

def discrete_cross_iou(xs, ys, width=30, img_shape=(720, 1280, 3)):
  xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
  ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]
  
  ious = np.zeros((len(xs), len(ys)))
  for i, x in enumerate(xs):
      for j, y in enumerate(ys):
          ious[i, j] = (x & y).sum() / (x | y).sum()
  return ious

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  whole_lane_test = []
  for test in bb_test:
    lane = []
    x1,y1,x2,y2,_ = test
    lane.append([x1,y1])
    lane.append([x2,y2])
    whole_lane_test.append(lane)

  whole_lane_gt = []
  for gt in bb_gt:
    lane = []
    x1,y1,x2,y2,_ = gt
    lane.append([x1,y1])
    lane.append([x2,y2])
    whole_lane_gt.append(lane)   
  
  o = discrete_cross_iou(whole_lane_test, whole_lane_gt)
  # print(o.shape)                                          
  return(o)  

# considering realtime run
# def iou_batch(bb_test, bb_gt):
#   """
#   From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#   """
#   test = bb_test[:,0] + (540 - bb_test[:,1]) / (bb_test[:,3] - bb_test[:,1]) * (bb_test[:,2] - bb_test[:,0])
#   gt = bb_gt[:,0] + (540 - bb_gt[:,1]) / (bb_gt[:,3] - bb_gt[:,1]) * (bb_gt[:,2] - bb_gt[:,0])
  
#   def L1(k, l):
#     return abs(k - l)

#   o = np.zeros((len(test),len(gt)))
#   for i in range(len(test)):
#     for j in range(len(gt)):
#       o[i][j] = 1 - L1(test[i], gt[j]) / 1280
#   # print(o)                                          
#   return(o)  

# seq = "2023-05-25-15-30-57"
def evaluate(seq):
  gt_path = "/data1/pbw/Mydataset/label/"
  gt_path = os.path.join(gt_path, seq)
  track_path = "/data1/pbw/Project/sort/output"
  track_path = os.path.join(track_path, '%s.txt'%(seq))
  iou_threshold = 0.5

  seq_dets = np.loadtxt(track_path, delimiter=',')
  frames = sorted(list(set(seq_dets[:, 0])))
  true_num = 0
  false_num = 0
  first = True
  correspond = {}
  for frame in frames:
      detections = seq_dets[seq_dets[:, 0]==frame, 2:7]
      track_ids = seq_dets[seq_dets[:, 0]==frame, 1]
      frame = ("%.6f" % frame)
      frame_name = str(frame) + '.json'
      frame = float(frame)
      json_path = os.path.join(gt_path, frame_name)
      gts = []
      gt_ids = []
      with open(json_path, 'r') as f:
          result = json.load(f)
          dataList = result.get('dataList')
          for dict in dataList:
              if dict['shapeType'] != 'line':
                continue
              id = dict['label']
              id = int(id[-1])
              coordinates = dict['coordinates']
              x1, y1, x2, y2 = coordinates[0][0], coordinates[0][1],coordinates[-1][0], coordinates[-1][1]
              lane = np.array([x1, y1, x2, y2, 1])
              gts.append(lane)
              gt_ids.append(id)
      gts = np.array(gts)
      iou_matrix = iou_batch(detections, gts)
      if min(iou_matrix.shape) > 0:
          a = (iou_matrix > iou_threshold).astype(np.int32)
          if a.sum(1).max() == 1 and a.sum(0).max() == 1:
              matched_indices = np.stack(np.where(a), axis=1)
          else:
              matched_indices = linear_assignment(-iou_matrix)
      else:
          matched_indices = np.empty(shape=(0,2))

      for matched_indice in matched_indices:
          x, y = matched_indice
          track_id = track_ids[x]
          gt_id = gt_ids[y]
          if gt_id not in correspond:
              correspond.setdefault(gt_id, [track_id])
              true_num += 1
          else:
              if track_id in correspond[gt_id]:
                true_num += 1
              else:
                false_num += 1
                correspond[gt_id].append(track_id)

  print(f"true_num: {true_num}")
  print(f"false_num: {false_num}")
  print(f"success rate: {true_num / (true_num + false_num)}")

