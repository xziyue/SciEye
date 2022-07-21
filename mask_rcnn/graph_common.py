import os
import sys

_file_path = __file__
_file_dir = os.path.dirname(_file_path)
project_dir = os.path.split(_file_dir)[0]
mask_rcnn_dir = os.path.join(project_dir, 'mask_rcnn')

tensorpack_path = os.path.join(project_dir, 'tensorpack')
faster_rcnn_path = os.path.join(tensorpack_path, 'examples', 'FasterRCNN')


sys.path.append(project_dir)
sys.path.insert(0,faster_rcnn_path)

# reorder inclusion path
path1, path2 = [], []
for path in sys.path:
    if '.local' in path:
        path2.append(path)
    else:
        path1.append(path)

sys.path = path1

print('include paths:', sys.path)