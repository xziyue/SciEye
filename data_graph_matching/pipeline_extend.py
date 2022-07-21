import os
from google.cloud import vision
import io
from data_graph_matching.pipeline import pipeline_helper, cprint


def detect_text(path):
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    data = []

    for text in texts:
        item = dict()
        item['t'] = text.description
        item['v'] = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        data.append(item)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return data

def ppl_google_text_ocr(batch_config):
    _STEP_NAME = 'google_get_text_ocr'

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        return [batch_config['filenames'][i] for i in jmp_proceed]


    def step_func(image_filenames):
        ret = []
        for fn in image_filenames:
            assert os.path.exists(fn)
            try:
                text = detect_text(fn)
                ret.append((True, text))
            except Exception as e:
                ret.append((False, repr(e)))
        return ret

    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))

        for ind, jmp_ind in enumerate(jmp_proceed):
            ret = return_of_step_func[ind]
            if ret[0]:
                result[jmp_ind] = ret[1]
            else:
                detail_logs[jmp_ind] = ret[1]    
        
        result_dict['result'] = result
        log_dict['detail'] = detail_logs


    cprint('using Google OCR API for text detection', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        step_func,
        result_log_func
    )