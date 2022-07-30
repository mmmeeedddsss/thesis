import glob
import os
from subprocess import Popen

for filename in glob.iglob('' + '**/*.py', recursive=True):
    if 'run_all' not in filename:
        devnull = open(os.devnull, 'wb')
        Popen(['nohup', 'python', filename], stdout=devnull, stderr=devnull)
        print(filename)