from mask_rcnn.graph_common import *

import os
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import DatasetSplit, DatasetRegistry
from data_graph_matching.param import *
import skimage.io as skio
import skimage.transform as sktf
from tqdm import tqdm
import joblib
import json
import copy



class GraphDatasetFromJSON(DatasetSplit):

    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def training_roidbs(self):
        ret = []
        for item in self.data:
            new_item = copy.copy(item)
            new_item['boxes'] = np.asarray(item['boxes'], dtype=np.float32)
            new_item['class'] = np.asarray(item['class'], dtype=np.int32)
            new_item['is_crowd'] = np.asarray(item['is_crowd'], dtype=np.int8)
            new_item['segmentation'] = [[np.asarray(x, dtype=np.float32)] for x in item['segmentation']]
            ret.append(new_item)
        return ret

def register_graph_dataset():
    dataset_info_path = os.path.join(mask_rcnn_dir, 'dataset_tp_info.json')
    assert os.path.exists(dataset_info_path)
    with open(dataset_info_path, 'r') as infile:
        dataset_info = json.load(infile)

    for key in ['train', 'val', 'test']:
        name = 'graph_'+key
        val = dataset_info[key]
        DatasetRegistry.register(name, lambda x=val:GraphDatasetFromJSON(x))
        DatasetRegistry.register_metadata(name, "class_names", dataset_info['metadata']['label_names'])


if __name__ == '__main__':

    # ds = GraphDatasetGenerator(selected_labels=[1, 6], label_names=['graph_content', 'colorbar'])
    # all_data = ds.generate_all()

    # with open(os.path.join(binary_path, 'dataset_tp_info.json'), 'w') as outfile:
    #     json.dump(all_data, outfile, indent=1)


    import cv2
    from viz import draw_annotation
    dataset_info_path = os.path.join(mask_rcnn_dir, 'dataset_tp_info.json')
    assert os.path.exists(dataset_info_path)
    with open(dataset_info_path, 'r') as infile:
        dataset_info = json.load(infile)    

    roidbs = GraphDatasetFromJSON(dataset_info['train']).training_roidbs()
    print('data loaded')
    # for r in roidbs[:2]:
    #     im = cv2.imread(r["file_name"])
    #     vis = draw_annotation(im, r["boxes"], r["class"])
    #     cv2.imwrite(os.path.join(binary_path, 'test_ann.png'), vis)

    
    # basedir = '~/data/balloon'
    # roidbs = BalloonDemo(basedir, "train").training_roidbs()
    # print("#images:", len(roidbs))

    # from viz import draw_annotation
    # from tensorpack.utils.viz import interactive_imshow as imshow
    # import cv2
    # for r in roidbs:
    #     im = cv2.imread(r["file_name"])
    #     vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
    #     imshow(vis)