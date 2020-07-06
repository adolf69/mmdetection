# -*- coding:utf-8 -*-
# @author :adolf
import json

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class OcrTextDataset(CustomDataset):
    CLASSES = ('text',)

    def load_annotations(self, ann_file):
        data_infos = list()
        with open(ann_file, 'r') as fp:
            for img_info in fp.readlines():
                img_info_ = img_info.strip()
                img_info_dict = json.loads(img_info_)
                # print(img_info_dict)
                data_infos.append(img_info_dict)
        # data_infos.append(
        #     dict(
        #         filename=ann_list[i + 1],
        #         width=width,
        #         height=height,
        #         ann=dict(
        #             bboxes=np.array(bboxes).astype(np.float32),
        #             labels=np.array(labels).astype(np.int64))
        #     ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
