import os
import json
import logging

import numpy as np
from tqdm import tqdm

# import utils.culane_metric as culane_metric
import utils.openlane_metric as culane_metric

from .lane_dataset_loader import LaneDatasetLoader

SPLIT_FILES = {
    'train': "list/train.txt",
    'val': 'list/test.txt',
    'test': "list/new_normal.txt"
}


class MyDataset(LaneDatasetLoader):
    def __init__(self, max_lanes=None, split='train', root=None, official_metric=True):
        self.split = split
        self.root = root
        self.official_metric = official_metric
        self.logger = logging.getLogger(__name__)

        if root is None:
            raise Exception('Please specify the root directory')
        if split not in SPLIT_FILES:
            raise Exception('Split `{}` does not exist.'.format(split))
        
        self.anno_root = os.path.join(root, 'label')
        self.image_root = os.path.join(root, "images")
        self.list = os.path.join(self.root, SPLIT_FILES[split])

        self.img_w, self.img_h = 1280, 720
        self.annotations = []
        self.load_annotations()
        self.max_lanes = 4 if max_lanes is None else max_lanes

    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def get_metrics(self, raw_lanes, idx):
        lanes = []
        pred_str = self.get_prediction_string(raw_lanes)
        for lane in pred_str.split('\n'):
            if lane == '':
                continue
            lane = list(map(float, lane.split()))
            lane = [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
            lanes.append(lane)
        anno = culane_metric.load_culane_img_data(self.annotations[idx]['path'].replace('.jpg', '.lines.txt'))
        _, fp, fn, ious, matches = culane_metric.culane_metric(lanes, anno)

        return fp, fn, matches, ious

    def load_annotation(self, json_path):
        lanes = []
        with open(json_path, 'r') as file:
            # file_lines = [line for line in file]
            # info_dict = json.loads(file_lines[0])
            result = json.load(file)
            dataList = result.get('dataList')
            filePath = json_path
            img_name = filePath.split('/',5)[5].replace('json','png')
            image_path = os.path.join(self.image_root, img_name)
            lanes1 = dataList
            for idx in range(len(lanes1)):
                ori_lane = lanes1[idx]
                if ori_lane['shapeType'] != "line":
                    continue
                uv1 = np.array(ori_lane['coordinates'])
                uv1 = uv1.T
                lane = [(int(uv1[0][j]), int(uv1[1][j])) for j in range(len(uv1[1]) - 1) if uv1[0][j] >= 0 and uv1[1][j] >= 0]
                lanes.append(lane)
            # img_name = img_name.split('/',2)[2]
        assert os.path.exists(image_path), '{:s} not exist'.format(image_path)
        lanes = [lane for lane in lanes if len(lane) >= 2]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y

        return {'path': image_path, 'lanes': lanes}

    def load_annotations(self):
        self.annotations = []
        self.max_lanes = 0
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/mydataset_{}.json'.format(self.split)

        if os.path.exists(cache_path):
            self.logger.info('Loading Mydataset annotations (cached)...')
            with open(cache_path, 'r') as cache_file:
                data = json.load(cache_file)
                self.annotations = data['annotations']
                self.max_lanes = data['max_lanes']
        else:
            self.logger.info('Loading Mydataset annotations and caching...')
            with open(self.list, 'r') as list_file:
                for file in tqdm(list_file):
                    file = file.split()
                    json_line = file[0]
                    json_path = os.path.join(self.anno_root, json_line)
                    json_path = json_path.replace('png', 'json')
                    anno = self.load_annotation(json_path)
                    org_path = os.path.join('data', json_line)
                    anno['org_path'] = org_path

                    if len(anno['lanes']) > 0:
                        self.max_lanes = max(self.max_lanes, len(anno['lanes']))
                    self.annotations.append(anno)
            with open(cache_path, 'w') as cache_file:
                json.dump({'annotations': self.annotations, 'max_lanes': self.max_lanes}, cache_file)

        self.logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.annotations),
                         self.max_lanes)

    # def get_prediction_string(self, pred):
    #     ys = np.arange(self.img_h) / self.img_h
    #     out = []
    #     for lane in pred:
    #         xs = lane(ys)
    #         valid_mask = (xs >= 0) & (xs < 1)
    #         xs = xs * self.img_w
    #         lane_xs = xs[valid_mask]
    #         lane_ys = ys[valid_mask] * self.img_h
    #         lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
    #         lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
    #         if lane_str != '':
    #             out.append(lane_str)

    #     return '\n'.join(out)
    def get_prediction_string(self, pred, pred_path):
        ys = np.arange(self.img_h) / self.img_h
        lane_lines = []
        result = {}
        for lane in pred:
            uv_orgsize = []
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs[valid_mask]
            # xs = xs * self.cfg.ori_img_w
            lane_xs = xs * self.img_w
            lane_ys = ys[valid_mask] * self.img_h
            # lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            uv_orgsize.append(list(lane_xs))
            uv_orgsize.append(list(lane_ys))
            lane_lines.append({'uv': uv_orgsize})
        result['lane_lines'] = lane_lines
        
        with open(pred_path, 'w') as result_file:
            json.dump(result, result_file)

    def eval_predictions(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.annotations[idx]['old_anno']['org_path']))
            output_filename = os.path.basename(self.annotations[idx]['old_anno']['org_path']).replace(".png", ".json")
            os.makedirs(output_dir, exist_ok=True)
            pred_path = os.path.join(output_dir, output_filename)
            self.get_prediction_string(pred, pred_path)
            # output = self.get_prediction_string(pred)
            # with open(os.path.join(output_dir, output_filename), 'w') as out_file:
            #     out_file.write(output)
        return culane_metric.eval_predictions(output_basedir, self.anno_root, self.list, official=self.official_metric)

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
