import os
import subprocess
import platform

os_name = platform.system().lower()
zipfile = 'assignment_2_part_2_submission.zip'
if "windows" in os_name:
    if os.path.exists('./' + zipfile):
        command = ['del', zipfile]
        subprocess.run(command)

    command = ['tar', '-a', '-c', '-f', zipfile,
        'configs', 'losses', 'main.py', 'models', 'checkpoints']
    subprocess.run(command)
else:
    command = ['rm', '-f', zipfile]
    subprocess.run(command)

    command = ['zip', '-r', zipfile,
        'configs/', 'losses/', 'main.py', 'models/', 'checkpoints/']
    subprocess.run(command)
 