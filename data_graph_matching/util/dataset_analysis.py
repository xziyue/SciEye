# analyzes the files in the dataset
import os
import json
import pandas

_my_path = os.path.dirname(__file__)
proj_path, _ = os.path.split(_my_path)
proj_path = os.path.dirname(proj_path)
binary_path = os.path.join(proj_path, 'binary_files')

dataset_root_path = os.path.join(proj_path, 'data2graph_competition')
dataset_data_path = os.path.join(dataset_root_path, 'data')

dataset_info_filename = os.path.join(binary_path, 'dataset_info.json')

