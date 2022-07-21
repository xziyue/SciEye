from termcolor import cprint
import json
import os
from data_graph_matching.util.dataset_analysis import proj_path, binary_path
from data_graph_matching.common_util import *
import numpy as np
import joblib
import itertools
from data_graph_matching.param import parallel_n_jobs
import copy



def pipeline_helper(
    batch_config,
    step_name,
    preprocessing_func, # signature: (batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop)
    step_func, # signature: (result_of_preprocessing_func)
    result_log_func, # signature: (result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop) adds additional results to batch_config
    dependent_steps=None,  # list of steps
    process_criteria=None, # signature: (batch_config, results_of_dependent_steps) returns: (True/False, detail message)
    ):

    _STEP_NAME = step_name
    log_stack = batch_config['logs']
    result_stack = batch_config['results']

    log_dict = dict(step_name=_STEP_NAME)
    result_dict = dict(step_name=_STEP_NAME)

    dependent_step_results = []

    if dependent_steps is not None:
        for step_name in dependent_steps:
            found_dependency = False
            for res_dict in batch_config['results']:
                if step_name in res_dict['step_name']:
                    dependent_step_results.append(res_dict['result'])
                    found_dependency = True
                    break
        
            if not found_dependency:
                raise RuntimeError(f'error occurred in pipeline step {_STEP_NAME}:\n unable to find dependent step "{step_name}" in the pipeline')

    
    jmp_proceed = []
    jmp_stop = []

    if process_criteria is not None:
        criteria_results = process_criteria(batch_config, dependent_step_results)
        criteria_log = []
        for ind, (valid, details) in enumerate(criteria_results):
            criteria_log.append(details)
            if valid:
                jmp_proceed.append(ind)
            else:
                jmp_stop.append(ind)
        log_dict['process_criteria_log'] = criteria_log
    else:
        jmp_proceed = list(range(len(batch_config['filenames'])))


    input_to_step_func = preprocessing_func(batch_config, dependent_step_results, jmp_proceed, jmp_stop)
    if isinstance(input_to_step_func, tuple):
        assert len(input_to_step_func[0]) == len(jmp_proceed)
        ret_step_func = step_func(*input_to_step_func)
    else:
        assert len(input_to_step_func) == len(jmp_proceed)
        ret_step_func = step_func(input_to_step_func)

    result_log_func(result_dict, log_dict, ret_step_func, jmp_proceed, jmp_stop)

    result_stack.append(result_dict)
    log_stack.append(log_dict)


# def ppl_batch_google_text_ocr(batch_config):
#     raise RuntimeError('this method is not implemented')


def ppl_batch_get_graph_component(batch_config):
    from data_graph_matching.graph_component_detection import get_graph_component
    _STEP_NAME = 'get_graph_component'

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        return [batch_config['filenames'][i] for i in jmp_proceed]

    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        log_dict['stdout'] = return_of_step_func['stdout']
        log_dict['stderr'] = return_of_step_func['stderr']

        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))
        if 'result' in return_of_step_func:
            for ind, jmp_ind in enumerate(jmp_proceed):
                item = return_of_step_func['result'][ind]
                
                if 1 in item['class_ids']:
                    one_index = item['class_ids'].index(1)
                    box = item['boxes'][one_index]
                    j_min, i_min, j_max, i_max = np.round(box).astype(np.int)
                    if i_max > i_min and j_max > j_min:
                        detail_logs[jmp_ind] = 'found figure content in detection step'
                        if result[jmp_ind] is None:
                            result[jmp_ind] = dict()
                        result[jmp_ind]['figure_content'] = box.tolist()
                    else:
                        detail_logs[jmp_ind] = 'figure content result is invalid'
                if 2 in item['class_ids']:
                    two_index = item['class_ids'].index(2)
                    box = item['boxes'][two_index]
                    detail_logs[jmp_ind] = detail_logs[jmp_ind] + ', ' if detail_logs[jmp_ind] is not None else '' + \
                                           'found color bar in detection step'
                    if result[jmp_ind] is None:
                        result[jmp_ind] = dict()
                    result[jmp_ind]['color_bar'] = box.tolist()
                
                if result[jmp_ind] is None:
                    detail_logs[jmp_ind] = 'cannot find figure content in detection step'
        
        result_dict['result'] = result
        log_dict['detail'] = detail_logs


    cprint('localizing main figure content and color bar', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        get_graph_component,
        result_log_func
    )




# align the text information in the image
def ppl_get_text_alignment(batch_config):
    from data_graph_matching.image_operation import ImageOp
    _STEP_NAME = 'get_text_alignment'

    dependent_steps = ['get_text_ocr', 'get_graph_component']

    def process_criteria(batch_config, results_of_dependent_steps):
        bools = []
        msgs = []
        for i in range(len(batch_config['filenames'])):
            if results_of_dependent_steps[0][i] is None or results_of_dependent_steps[1][i] is None:
                bools.append(False)
                msgs.append(f'result from one of the dependent step is missing')
            else:
                bools.append(True)
                msgs.append('OK')
        return zip(bools, msgs)

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        image_infos = []
        for jmp_ind in jmp_proceed:
            image_infos.append(
                dict(
                    filename=batch_config['filenames'][jmp_ind],
                    text=results_of_dependent_steps[0][jmp_ind],
                    figure_content_box=results_of_dependent_steps[1][jmp_ind]['figure_content']
                )
            )
        return image_infos
    
    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))

        for ind, jmp_ind in enumerate(jmp_proceed):
            clusters, unclustered = return_of_step_func[ind]
            for cls in clusters:
                cls['text'] = [x.to_dict() for x in cls['text']]
                cls['position'] = cls['position'].value
            unclustered_dict = unclustered
            detail_logs[jmp_ind] = 'OK'
            result[jmp_ind] = dict(
                cluster=clusters,
                unclustered=unclustered_dict
            )
        
        for ind, jmp_ind in enumerate(jmp_stop):
            detail_logs[jmp_ind] = 'text alignment failed (maybe a dependent step failed?)'
        
        result_dict['result'] = result
        log_dict['detail'] = detail_logs

    def apply_analysis(info):
        img_op = ImageOp(info)
        return img_op.get_text_alignment_threshold_based()

    def step_func(image_infos):
        return joblib.Parallel(n_jobs=parallel_n_jobs, pre_dispatch='all')(joblib.delayed(apply_analysis)(x) for x in image_infos)

    cprint('performing text alignment', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        step_func,
        result_log_func,
        process_criteria=process_criteria,
        dependent_steps=dependent_steps
    )


# align the text information in the image
def ppl_get_legend_color(batch_config):
    from data_graph_matching.image_operation import ImageOp, TextPosAgainstFigureContent
    _STEP_NAME = 'get_legend_color'

    dependent_steps = ['get_text_ocr', 'get_text_alignment']

    def process_criteria(batch_config, results_of_dependent_steps):
        bools = []
        msgs = []
        for i in range(len(batch_config['filenames'])):
            if results_of_dependent_steps[0][i] is None or results_of_dependent_steps[1][i] is None:
                bools.append(False)
                msgs.append(f'result from one of the dependent step is missing')
            else:
                # check if there is a text group to the right of the main figure content
                for item in results_of_dependent_steps[1][i]['cluster']:
                    found_right_group = False
                    if item['position'] == TextPosAgainstFigureContent.RIGHT.value:
                        found_right_group = True
                        # check if every value is numeric, if so, it is likely to be a colorbar...
                        all_text = [is_numeric_text(x['text']['t']) for x in item['text']]
                        if all(all_text):
                            bools.append(False)
                            msgs.append('the text group to the right of figure content is likely to be from color bar')
                        else:
                            bools.append(True)
                            msgs.append('OK')

                        break

                if not found_right_group:
                    text_cluster_result = results_of_dependent_steps[1][i]
                    found_dummy = False
                    for item in text_cluster_result['unclustered']:
                        if TextPosAgainstFigureContent.RIGHT.value in item['position']:
                            found_dummy = True
                            break

                    if found_dummy:
                            bools.append(True)
                            msgs.append('no text group on the right is found, adding dummy group')
                    else:
                        bools.append(False)
                        msgs.append('this figure does not have a text group to the right of figure content')
        return zip(bools, msgs)

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        image_infos = []
        for jmp_ind in jmp_proceed:
            target_group = None
            for item in results_of_dependent_steps[1][jmp_ind]['cluster']:
                if item['position'] == TextPosAgainstFigureContent.RIGHT.value:
                    target_group = item
                    break

            if target_group is None:
                text_cluster_result = results_of_dependent_steps[1][jmp_ind]
                dummy_cluster = []
                for item in text_cluster_result['unclustered']:
                    if TextPosAgainstFigureContent.RIGHT.value in item['position']:
                        dummy_cluster.append({
                            'text': [item],
                            'index': 0,
                            'vertex_index': 0,
                            'coordinate_index': 0,
                            'position': TextPosAgainstFigureContent.RIGHT.value
                        })
                        break
                assert len(dummy_cluster) > 0
                target_group = dummy_cluster[0]

            assert target_group is not None

            image_infos.append(
                dict(
                    filename=batch_config['filenames'][jmp_ind],
                    text=results_of_dependent_steps[0][jmp_ind],
                    #figure_content_box=results_of_dependent_steps[1][jmp_ind],
                    legend_text_group=target_group
                )
            )
        return image_infos
    
    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))

        for ind, jmp_ind in enumerate(jmp_proceed):
            res = return_of_step_func[ind]
            if len(res) > 0:
                result[jmp_ind] = res
                detail_logs[jmp_ind] = 'OK'
            else:
                detail_logs[jmp_ind] = 'no color for this legend is found'
        
        for ind, jmp_ind in enumerate(jmp_stop):
            detail_logs[jmp_ind] = 'cannot get color for legend'
        
        result_dict['result'] = result
        log_dict['detail'] = detail_logs

    def apply_analysis(info):
        img_op = ImageOp(info)
        return img_op.get_legend_colors(info['legend_text_group'])

    def step_func(image_infos):
        return joblib.Parallel(n_jobs=parallel_n_jobs, pre_dispatch='all')(joblib.delayed(apply_analysis)(x) for x in image_infos)

    cprint('finding color for legends', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        step_func,
        result_log_func,
        process_criteria=process_criteria,
        dependent_steps=dependent_steps
    )


def ppl_get_figure_content_classification_new(batch_config):
    from data_graph_matching.graph_content_classification_new import get_graph_content_classification_new
    log_dict, result_dict = get_graph_content_classification_new(batch_config)
    batch_config['logs'].append(log_dict)
    batch_config['results'].append(result_dict)


def ppl_fine_tune_figure_region(batch_config):
    _STEP_NAME = 'fine_tune_figure_region'
    dependent_steps = ['get_graph_component', 'get_legend_color']

    def process_criteria(batch_config, results_of_dependent_steps):
        bools = []
        msgs = []
        for i in range(len(batch_config['filenames'])):
            if results_of_dependent_steps[0][i] is None:
                bools.append(False)
                msgs.append(f'no results found in dependent step {dependent_steps[0]}')
            else:
                bools.append(True)
                msgs.append('OK')
        return zip(bools, msgs)

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        boxes = [results_of_dependent_steps[0][i]['figure_content'] for i in jmp_proceed]
        all_legend_j = [results_of_dependent_steps[1][i] for i in jmp_proceed]
        for i in range(len(boxes)):
            # use legend information to fine-tune figure content localization
            j_min, i_min, j_max, i_max = np.round(boxes[i]).astype(np.int)
            if all_legend_j[i] is None:
                continue
            all_j_mins = [x[2]['j_min'] for x in all_legend_j[i]]
            j_max_ft = min(j_max, min(all_j_mins))
            if j_max_ft > j_min:
                boxes[i] = np.asarray((j_min, i_min, j_max_ft, i_max)).astype(np.int).tolist()
        return boxes

    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))

        for ind, jmp_ind in enumerate(jmp_proceed):
            res = return_of_step_func[ind]
            detail_logs[jmp_ind] = 'OK'
            result[jmp_ind] = res
        
        for ind, jmp_ind in enumerate(jmp_stop):
            detail_logs[jmp_ind] = 'figure content location fine tuning failed'
        
        result_dict['result'] = result
        log_dict['detail'] = detail_logs


    cprint('fine tuning figure region localization', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        lambda x : x,
        result_log_func,
        process_criteria=process_criteria,
        dependent_steps=dependent_steps
    )

# drop disqualified images along the way
def ppl_batch_get_figure_content_classification(batch_config):
    from data_graph_matching.graph_content_classification import get_figure_content_classification, model_labels
    _STEP_NAME = 'figure_content_classification'

    dependent_steps = ['fine_tune_figure_region']

    def process_criteria(batch_config, results_of_dependent_steps):
        bools = []
        msgs = []
        for i in range(len(batch_config['filenames'])):
            if results_of_dependent_steps[0][i] is None:
                bools.append(False)
                msgs.append(f'no results found in dependent step')
            else:
                bools.append(True)
                msgs.append('OK')
        return zip(bools, msgs)

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        filenames = [batch_config['filenames'][i] for i in jmp_proceed]
        boxes = [results_of_dependent_steps[0][i] for i in jmp_proceed]
        return (filenames, boxes)

    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))
        logits = [None] * (len(jmp_proceed) + len(jmp_stop))

        for ind, jmp_ind in enumerate(jmp_proceed):
            res = return_of_step_func[ind]
            logits[jmp_ind] = res.tolist()
            detail_logs[jmp_ind] = 'OK'
            assert len(res) == len(model_labels)
            result[jmp_ind] = model_labels[np.argmax(res, axis=-1)]
        
        for ind, jmp_ind in enumerate(jmp_stop):
            detail_logs[jmp_ind] = 'figure content classification failed (maybe a dependent step failed?)'
        
        result_dict['result'] = result
        log_dict['detail'] = detail_logs
        log_dict['logit'] = logits

 
    cprint('classifying figure content', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        get_figure_content_classification,
        result_log_func,
        process_criteria=process_criteria,
        dependent_steps=dependent_steps
    )


# align the text information in the image
def ppl_parse_graph(batch_config):
    from data_graph_matching.graph_understanding import GraphParser
    _STEP_NAME = 'parse_graph'

    dependent_steps = ['get_text_alignment', 'get_legend_color', 'get_graph_component',
                       'fine_tune_figure_region', 'figure_content_classification']

    normal_steps = [0, 1, 3, 4]
    heatmap_steps = [0, 2, 4]

    def process_criteria(batch_config, results_of_dependent_steps):
        bools = []
        msgs = []
        for i in range(len(batch_config['filenames'])):
            good = True

            if results_of_dependent_steps[-1][i] == 'heatmap':
                steps = heatmap_steps
            else:
                steps = normal_steps

            for j in range(len(steps)):
                if results_of_dependent_steps[steps[j]][i] is None:
                    bools.append(False)
                    msgs.append(f'result from dependent step "{steps[j]}" is missing')
                    good = False
                    break

            if good:
                bools.append(True)
                msgs.append('OK')

        return zip(bools, msgs)

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        image_infos = []
        for jmp_ind in jmp_proceed:
            fig_type = results_of_dependent_steps[-1][jmp_ind]
            if fig_type == 'heatmap':
                image_infos.append(
                    (
                        batch_config['filenames'][jmp_ind], # filename
                        results_of_dependent_steps[4][jmp_ind], # graph type
                        results_of_dependent_steps[2][jmp_ind]['figure_content'], # figure content region
                        results_of_dependent_steps[0][jmp_ind], # text alignment
                        None, # legend
                    )
                )
            else:
                image_infos.append(
                    (
                        batch_config['filenames'][jmp_ind], # filename
                        results_of_dependent_steps[4][jmp_ind], # graph type
                        results_of_dependent_steps[3][jmp_ind], # figure content region
                        results_of_dependent_steps[0][jmp_ind], # text alignment
                        results_of_dependent_steps[1][jmp_ind], # legend
                    )
                )
        return image_infos
    
    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))

        for ind, jmp_ind in enumerate(jmp_proceed):
            success, info, config = return_of_step_func[ind]
            if success:
                result[jmp_ind] = config
            else:
                detail_logs[jmp_ind] = info
        
        for ind, jmp_ind in enumerate(jmp_stop):
            detail_logs[jmp_ind] = 'graph parsing failed (maybe a dependent step failed?)'
        
        result_dict['result'] = result
        log_dict['detail'] = detail_logs

    def apply_analysis(info):
        parser = GraphParser()
        parser.from_pipeline(*info)
        success, info = parser.parse()
        return success, info, parser.config

    def step_func(image_infos):
        return joblib.Parallel(n_jobs=parallel_n_jobs, pre_dispatch='all')(joblib.delayed(apply_analysis)(x) for x in image_infos)

    cprint('parsing graph information', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        step_func,
        result_log_func,
        process_criteria=process_criteria,
        dependent_steps=dependent_steps
    )


# align the text information in the image
def ppl_match_dataset(batch_config):
    from data_graph_matching.graph_data_matching import GraphDatasetMatcher
    _STEP_NAME = 'match_dataset'

    dependent_steps = ['parse_graph']

    def process_criteria(batch_config, results_of_dependent_steps):
        bools = []
        msgs = []
        for i in range(len(batch_config['filenames'])):
            good = True
            for j in range(len(dependent_steps)):
                if results_of_dependent_steps[j][i] is None:
                    bools.append(False)
                    msgs.append(f'result from dependent step "{dependent_steps[j]}" is missing')
                    good = False
                    break
            if good:
                bools.append(True)
                msgs.append('OK')
        return zip(bools, msgs)

    def preprocessing_func(batch_config, results_of_dependent_steps, jmp_proceed, jmp_stop):
        return [results_of_dependent_steps[0][i] for i in jmp_proceed]

    def result_log_func(result_dict, log_dict, return_of_step_func, jmp_proceed, jmp_stop):
        result = [None] * (len(jmp_proceed) + len(jmp_stop))
        detail_logs = [''] * (len(jmp_proceed) + len(jmp_stop))

        for ind, jmp_ind in enumerate(jmp_proceed):
            success, res = return_of_step_func[ind]
            if success:
                if isinstance(res, tuple) and res[0] is None:
                    detail_logs[jmp_ind] = res[1]
                else:
                    result[jmp_ind] = res
            else:
                detail_logs[jmp_ind] = res

        for ind, jmp_ind in enumerate(jmp_stop):
            detail_logs[jmp_ind] = 'graph parsing failed (maybe a dependent step failed?)'

        result_dict['result'] = result
        log_dict['detail'] = detail_logs

    def apply_analysis(res):
        try:
            matcher = GraphDatasetMatcher(res, batch_config['dataset_candidates'])
            result = matcher.match()
            return True, result
        except Exception as e:
            return False, repr(e)

    def step_func(image_infos):
        # don't use parallel for this step...
        return joblib.Parallel(n_jobs=1, pre_dispatch='all')(
            joblib.delayed(apply_analysis)(x) for x in image_infos)

    cprint('matching datasets', 'green')
    pipeline_helper(
        batch_config,
        _STEP_NAME,
        preprocessing_func,
        step_func,
        result_log_func,
        process_criteria=process_criteria,
        dependent_steps=dependent_steps
    )



def run_pipeline_batched_raw(steps, batch_configs):
    for batch_ind, batch_config in enumerate(batch_configs):
        cprint('running batch {}/{} (batch size={})'.format(
                batch_ind + 1, 
                len(batch_configs),
                len(batch_config['filenames'])), 'yellow')
        for step in steps:
            step(batch_config)
            #print(batch_config)
    return batch_configs


def run_pipeline_batched(steps, list_of_images, batch_size=32):
    
    batch_configs = []

    for i in range(0, len(list_of_images), batch_size):
        l = i
        r = min(i + batch_size, len(list_of_images))
        sl = list_of_images[l:r]
        
        batch_config = {
            'filenames' : sl,
            'results': [],
            'logs': []
        }

        batch_configs.append(batch_config)

    return run_pipeline_batched_raw(steps, batch_configs)

