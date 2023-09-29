import subprocess
import os

def run_system_cmd(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True)
    if p.returncode != 0:

        raise SystemError(str(p.stderr))
    else:
        print(str(p.stdout))
        
def encrypt_file(model_dir):
    run_system_cmd('bash ./detect_class_process/encrypt_model.sh {}'.format(model_dir))
    for _f in os.listdir(model_dir):
        if _f.endswith('encrypted'):
            _f_orig = '_'.join(_f.split('_')[:-1])
            run_system_cmd('mv {} {}'.format(os.path.join(model_dir, _f), 
                    os.path.join(model_dir, _f_orig)))
