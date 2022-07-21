import subprocess
import sys
import os
from data_graph_matching.util.dataset_analysis import proj_path
import tempfile
import copy
import joblib

_run_dir = os.path.join(proj_path, 'mask_rcnn')

def get_graph_component(list_of_images, cuda_device=(0,)):
    new_env = copy.copy(os.environ)
    new_env['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in cuda_device])
    new_env['PYTHONPATH'] = proj_path
    ret = dict()
    with tempfile.NamedTemporaryFile() as tmpf:
        cp = subprocess.run(
            ['bash', os.path.join(_run_dir,'predict.sh'), '-i', ' '.join(list_of_images), 
            '-o', tmpf.name],
            cwd=proj_path,
            env=new_env,
            capture_output=True
        )
        if os.stat(tmpf.name).st_size > 0:
            ret['result'] = joblib.load(tmpf.name)
        ret['stdout'] = cp.stdout.decode()
        ret['stderr'] = cp.stderr.decode()
        # ret['stdout'] = 'cp.stdout.decode()'
        # ret['stderr'] = 'cp.stderr.decode()'
        return ret

if __name__ == '__main__':
    ret = get_graph_component([os.path.join(proj_path, 'binary_files', 'graph_rendered_generation', 'area_2.png')])
    print(ret)