#!/usr/bin/python2

import os
import subprocess
import time
import random

def spec_cmd(spec_prog):
    return "taskset 0x8 runspec --config=test.cfg --size=train" \
    " --noreportable --tune=base --iterations=1 {spec_prog}".format(
        spec_prog=spec_prog)

spec_benchmarks = ('perlbench', 'bzip2', 'gcc', 'mcf', 'milc', 'namd',
'gobmk', 'soplex', 'povray', 'hmmer', 'sjeng', 'libquantum',
'h264ref', 'lbm', 'omnetpp', 'astar')

while True:

    running_processes = []
    for spec_prog in ['perlbench', 'gcc', 'libquantum']:
        cmd = spec_cmd(spec_prog)
        print(cmd)
        spec_process = subprocess.Popen(cmd.split())
        running_processes.append(spec_process)

    for p in running_processes:
        p.wait()
    time.sleep(1)
