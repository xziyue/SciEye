import sys
import os
from data_graph_matching.util.dataset_analysis import proj_path, binary_path
import tempfile
import numpy as np
import skimage
import skimage.io as skio
import skimage.color as skclr
import matplotlib.pyplot as plt
import copy
import multiprocessing


c_image_size = (224, 224)

model_labels = ['area', 'bar', 'box', 'hbar', 'heatmap', 'line', 'scatter', 'stacked', 'violin']

def read_image(img_fn, box, tf):
    img = skimage.img_as_float32(skio.imread(img_fn))
    if len(img.shape) == 2:
        img = skclr.gray2rgb(img)
    elif img.shape[2] == 4:
        img = skclr.rgba2rgb(img)
    img = 1.0 - img
    box = np.round(box).astype(np.int)
    j_min, i_min, j_max, i_max = box
    i_min, i_max = np.clip([i_min, i_max], 0, img.shape[0])
    j_min, j_max = np.clip([j_min, j_max], 0, img.shape[1])
    if i_max > i_min and j_max > j_min:
        img = img[i_min : i_max, j_min : j_max, ...]
    else:
        ci = img.shape[0] // 2
        cj = img.shape[1] // 2
        img = img[ci-ci//2:ci+ci//2, cj-cj//2:cj+cj//2]
    img = tf.convert_to_tensor(img)
    resized = tf.image.resize_with_pad(img, c_image_size[1], 
                                    c_image_size[0])
    resized = 1.0 - resized
    result = resized


    # gs_img = tf.image.rgb_to_grayscale(resized)
    # edges = tf.image.sobel_edges(tf.expand_dims(gs_img, axis=0))[0, ...]
    # sobel_intensity = tf.math.sqrt(edges[:, :, :, 0] ** 2 + edges[:, :, :, 1] ** 2)
    # result = tf.concat([resized, sobel_intensity], axis=-1)
    return result

def get_figure_content_classification_task(list_of_images, boxes, output, cuda_device):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in cuda_device])
    import tensorflow as tf

    assert len(list_of_images) == len(boxes)

    class DummyObject:
        def __init__(self):
            pass

        def from_config(self):
            pass

    def dummy_loss(a, b):
        return a
    
    model = tf.keras.models.load_model(os.path.join(proj_path, 'ckpt', 'model'),
        custom_objects={'MyLRSchedule2': DummyObject, 'compute_loss': dummy_loss})

    model.load_weights(os.path.join(proj_path, 'ckpt', 'cls-ckpt'))

    img_tensors = [read_image(x, y, tf) for x, y in zip(list_of_images, boxes)]
    all_img = tf.stack(img_tensors, axis=0)

    logits = model.predict(all_img)
    
    output.put(logits)


def get_figure_content_classification(list_of_images, boxes, cuda_device=(0,)):
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=get_figure_content_classification_task, args=(list_of_images, boxes, q, cuda_device))
    proc.start()
    proc.join()
    logits = q.get()
    return logits

if __name__ == '__main__':
    ret = get_figure_content_classification([os.path.join(proj_path, 'binary_files', 'graph_rendered_generation', 'area_1.png')])
    print('result:')
    print(ret)