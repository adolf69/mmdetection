# -*- coding:utf-8 -*-
# @author :adolf
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

image_dir = "/home/shizai/datadisk2/coco/val2017/"
annotations_file = "/home/shizai/datadisk2/coco/annotations/instances_val2017.json"
with open(annotations_file, 'r') as fp:
    line_json = json.load(fp)
    # print(line_json.keys())
    # print('1111', len(line_json['images']))
    # print('2222', len(line_json['annotations']))
    # print('3333',len(line_json['categories']))
    image_info = line_json['images']
    print(image_info[1])
    annotations = line_json['annotations'][2]
    image_id_ = annotations['image_id']
    # print(annotations['category_id'])
    annotations_categories = annotations['category_id']
    categories = line_json['categories']  # [annotations['category_id']]

# print(image_id_)
for one_anno in image_info:
    if int(one_anno['id']) == image_id_:
        # print(image_info)
        image_path: str = os.path.join(image_dir, one_anno['file_name'])
        break

print(image_path)
image = cv2.imread(image_path)
# print(image.shape)
# print(image_info)
# print(annotations)
for one_categories in categories:
    if int(one_categories['id']) == annotations_categories:
        print(one_categories['name'])
        break
bbox = annotations['bbox']
bbox = [int(point) for point in bbox]
cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1), (0, 0, 255), 2)

segmentation = annotations['segmentation']
print(segmentation)

segmentation_list = list()
for i in range(0, len(segmentation[0]), 2):
    point = [segmentation[0][i], segmentation[0][i + 1]]
    segmentation_list.append(point)

segmentation_arr = np.array(segmentation_list, dtype=np.int32)
# print(segmentation_arr)
cv2.fillPoly(image, [segmentation_arr], (0, 255, 0))

# cv2.imwrite('test.png', image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
