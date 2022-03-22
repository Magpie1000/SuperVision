import shlex
import subprocess
import os
import sys

abs_project_path = '/'.join(os.path.abspath(__file__).replace('\\','/').split('/')[:-2]) + '/'

# Run command in Shell
def run_shell(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

# Project Folders
def make_folder():
    os.makedirs(os.path.dirname(abs_project_path + 'Data/'), exist_ok=True)
    os.makedirs(os.path.dirname(abs_project_path + 'Model/'), exist_ok=True)
    os.makedirs(os.path.dirname(abs_project_path + 'Output/'), exist_ok=True)

# Get list of files from a folder
def get_file_list(folder): # folder is relative to project
    abs_path = abs_project_path + folder
    print(folder, abs_path)
    filenames = os.listdir(abs_path)
    return filenames

# Check if file exist or not
def does_not_exists(file):
    abs_path = abs_project_path + file
    if os.path.exists(abs_path):
        return False
    return True


def evaluate_vmaf_score(file_inp, file_out, width, height):
    if sys.platform == 'linux' or sys.platform == 'linux2':
        vmaf_path = abs_project_path + 'Utils/vmaf'
    else:
        vmaf_path = abs_project_path + 'Utils/vmaf.exe'
    reference_path = abs_project_path + file_inp
    distorted_path = abs_project_path + file_out

    # [SSAFY] shell command to get vmaf score
    cmd = f'{vmaf_path} -r {reference_path} -d {distorted_path} -w {width} -h {height} -p 444 -b 8 --feature psnr'
    # cmd=f'pwd'
    # print(cmd)
    run_shell(cmd)

# Run make_folder on import
make_folder()