# -*- coding:utf-8 -*-
# @author :adolf
import os
import cv2
import numpy as np
import json
from numpy.core._multiarray_umath import ndarray

file_dir = "data/icdar15/train/"
img_dir = os.path.join(file_dir, "imgs")
gt_dir = os.path.join(file_dir, "gts")

img_list = os.listdir(img_dir)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_box(one_mask):
    xmin = min(one_mask[0], one_mask[2])
    xmax = max(one_mask[5], one_mask[7])
    ymin = min(one_mask[1], one_mask[6])
    ymax = max(one_mask[3], one_mask[4])
    return [xmin, ymin, xmax, ymax]


anno_list = list()

for img_name in img_list:
    print(img_name)
    # if img_name != "img_893.jpg":
    #     continue

    img = cv2.imread(os.path.join(img_dir, img_name))
    height, width = img.shape[:2]
    # print(height,width)
    img_info_dict = dict(filename=img_name,
                         width=width,
                         height=height,
                         ann=dict(
                             bboxes=None,
                             labels=None,
                             masks=None)
                         )
    gt_name = img_name.replace('jpg', 'txt')
    gt_name = "gt_" + gt_name

    masks = list()
    labels = list()
    bboxes = list()

    with open(os.path.join(gt_dir, gt_name), 'r') as fp:
        for line in fp.readlines():
            line_list = line.strip().replace('\ufeff', '').split(',')
            # print(line_list)
            if line_list[-1] in ["###", "*"]:
                continue
            line_list = line_list[:8]
            line_list = list(map(float, line_list))
            # print(line_list)
            one_mask = line_list
            one_label = 0
            one_bbox = get_box(one_mask)
            # print(one_mask)
            masks.append(one_mask)
            labels.append(one_label)
            bboxes.append(one_bbox)
    masks_arr: ndarray = np.array(masks).astype(np.float32)
    labels_arr: ndarray = np.array(labels).astype(np.int64)
    bboxes_arr: ndarray = np.array(bboxes).astype(np.float32)

    img_info_dict['ann']['bboxes'] = bboxes_arr
    img_info_dict['ann']['labels'] = labels_arr
    img_info_dict['ann']['masks'] = masks_arr

    # print(img_info_dict)
    anno_list.append(img_info_dict)
    # break

with open('data/icdar15/annotation/icdar15_train.json', 'w') as fp:
    for ip in anno_list:
        fp.write(json.dumps(ip, cls=NumpyEncoder))
        fp.write('\n')
