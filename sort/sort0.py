"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from skimage import io
import sys
sys.path.append("/data1/pbw/Project")
import json
import cv2
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

def pred_produce(seq):
    global pred_path, lane_lines
    pred_path = "/data1/pbw/Project/save_results/lane_detection"
    pred_path = os.path.join(pred_path, seq)
    txt_save_path = "/data1/pbw/Project/sort/seq_det"
    txt_save_path = os.path.join(txt_save_path, '%s.txt'%(seq))
    with open(txt_save_path, 'w') as f1:
        for seg in sorted(os.listdir(pred_path)):
            json_path = os.path.join(pred_path, seg)
            with open(json_path, 'r') as f:
                result = json.load(f)
                lane_lines = result.get('lane_lines')
                # index = 0
                for dict in lane_lines:
                    uv = dict['uv']
                    x1, y1, x2, y2 = uv[0][0], uv[1][0] ,uv[0][-1], uv[1][-1]
                    if y1 > y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    id = seg.replace('.json','')
                    box_info = "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (id,-1, x1, y1, x2, y2, 1, -1, -1, -1)
                    f1.write(box_info)
                    f1.write('\n')
                    # index += 1

np.random.seed(0)

color_dict = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'black',
              6: 'purple', 7: 'white', 8: 'yellow', 9: 'brown', 10: 'cyan', 11: 'navy', 12: 'gold', 13: 'silver', 14: 'tan', 15: 'pink', 16: 'grey'}

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# def iou_batch(bb_test, bb_gt):
#   """
#   From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#   """
#   bb_gt = np.expand_dims(bb_gt, 0)
#   bb_test = np.expand_dims(bb_test, 1)
  
#   xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#   yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#   xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#   yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#   w = np.maximum(0., xx2 - xx1)
#   h = np.maximum(0., yy2 - yy1)
#   wh = w * h
#   o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)    
#   print(o.shape)                                          
#   return(o)  

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

# def iou_batch(bb_test, bb_gt):
#   """
#   From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#   """
#   whole_lane_test = []
#   for test in bb_test:
#     lane = []
#     x1,y1,x2,y2,_ = test
#     lane.append([x1,y1])
#     lane.append([x2,y2])
#     whole_lane_test.append(lane)

#   whole_lane_gt = []
#   for gt in bb_gt:
#     lane = []
#     x1,y1,x2,y2,_ = gt
#     lane.append([x1,y1])
#     lane.append([x2,y2])
#     whole_lane_gt.append(lane)   
  
#   o = discrete_cross_iou(whole_lane_test, whole_lane_gt)
#   # print(o.shape)                                          
#   return(o)  

# considering realtime run
def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  test = bb_test[:,0] + (540 - bb_test[:,1]) / (bb_test[:,3] - bb_test[:,1]) * (bb_test[:,2] - bb_test[:,0])
  gt = bb_gt[:,0] + (540 - bb_gt[:,1]) / (bb_gt[:,3] - bb_gt[:,1]) * (bb_gt[:,2] - bb_gt[:,0])
  
  def L1(k, l):
    return abs(k - l)

  o = np.zeros((len(test),len(gt)))
  for i in range(len(test)):
    for j in range(len(gt)):
      o[i][j] = 1 - L1(test[i], gt[j]) / 1280
  # print(o)                                          
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  # w = bbox[2] - bbox[0]
  # h = bbox[3] - bbox[1]
  # x = bbox[0] + w/2.
  # y = bbox[1] + h/2.
  # s = w * h    #scale is just area
  # r = w / float(h)
  w = bbox[0] - bbox[2]
  h = bbox[3] - bbox[1]
  x = bbox[0] - w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  # if(score==None):
  #   return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  # else:
  #   return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
  if(score==None):
    return np.array([x[0]+w/2.,x[1]-h/2.,x[0]-w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]+w/2.,x[1]-h/2.,x[0]-w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))
  # print(matched_indices)
  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      print(pos)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='/data1/pbw/Project/sort/seq_det/')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='2023-04-07-21-24-07')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=3)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display

  pred_produce(phase)

  if(display):
    if not os.path.exists('mydataset'):
      print('\n\tERROR: mydataset link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mydataset\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, '%s.txt'%(phase))
  # print(pattern)
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    id_dets = np.loadtxt(seq_dets_fn, dtype='str', delimiter=',', usecols=(0), unpack=True)
    # seq = seq_dets_fn[pattern.find('mydataset'):].split(os.path.sep)[0]
    seq = seq_dets_fn[pattern.find(phase):].split('.')[0]
    # print(seq)
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      # print(set(id_dets))
      # for frame in range(int(seq_dets[:,0].max())):
      frames = sorted(list(set(seq_dets[:, 0])))
      for frame in frames:
        # frame += 1 #detection and frame numbers begin at 1
        # print(str(seq_dets[:, 0]))
        # dets = seq_dets[str(seq_dets[:, 0])==frame[:-1], 2:7]
        # print(seq_dets)
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        # print(dets)
        # dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1
        # print(len(str(frame)))
        frame = ("%.6f" % frame)
        frame_name = str(frame) + '.png'
        frame = float(frame)
        if(display):
          fn = os.path.join('/data1/pbw/Mydataset/images', phase, frame_name)
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        # read oringinal json to show accurate lane
        frame_str = frame_name.replace('png','json')
        json_path = os.path.join(pred_path, frame_str)
        with open(json_path, 'r') as f:
            result = json.load(f)
        lane_lines = result.get('lane_lines')
        num = len(lane_lines)-1
        for d in trackers:
          print('%f,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2],d[3]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            dict = lane_lines[num]
            uv = dict['uv']
            ax1.add_line(lines.Line2D(uv[0],uv[1],linewidth=3,color=color_dict[d[4] % 15 + 1]))
            num -= 1  
        
        # for d in trackers:
        #   print('%f,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2],d[3]),file=out_file)
        #   if(display):
        #     d = d.astype(np.int32)
        #     ax1.add_line(lines.Line2D((d[0],d[2]),(d[1],d[3]),linewidth=3,color=color_dict[d[4] % 15 + 1]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / (total_time + 0.0001)))
  from evaluate import evaluate
  evaluate(phase)
  if(display):
    print("Note: to get real runtime results run without the option: --display")
