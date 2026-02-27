import os
import json

# seq = "2023-05-25-15-30-57"
def pred_produce(seq):
    pred_path = "/data1/pbw/Project/save_results/lane_detection"
    pred_path = os.path.join(pred_path, seq)
    txt_save_path = "/data1/pbw/Project/sort/seq_det"
    txt_save_path = os.path.join(txt_save_path, '%s.txt'%(seq))
    with open(txt_save_path, 'a+') as f1:
        for seg in sorted(os.listdir(pred_path)):
            json_path = os.path.join(pred_path, seg)
            with open(json_path, 'r') as f:
                result = json.load(f)
                lane_lines = result.get('lane_lines')
                for dict in lane_lines:
                    uv = dict['uv']
                    x1, y1, x2, y2 = uv[0][0], uv[1][0] ,uv[0][-1], uv[1][-1]
                    if y1 > y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    id = seg.replace('.json','')
                    box_info = "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (id,-1, x1, y1, x2, y2, 1, -1, -1, -1)
                    f1.write(box_info)
                    f1.write('\n')

