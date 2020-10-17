#!/usr/bin/python2

import os
import subprocess
import time

def spec_cmd(spec_prog):
    return "taskset 0x1 runspec --config=test.cfg --size=train"
    " --noreportable --tune=base --iterations=1 {spec_prog}".format(
        spec_prog=spec_prog)

while True:
    for spec_prog in ('perlbench', 'bzip2', 'gcc', 'mcf', 'milc', 'namd',
    'gobmk', 'soplex', 'povray', 'hmmer', 'sjeng', 'libquantum',
    'h264ref', 'lbm', 'omnetpp', 'astar'):
        cmd = spec_cmd(spec_prog)
        print(cmd)
        spec_process = subprocess.Popen(cmd.split())
        spec_status = spec_process.wait()
        time.sleep(1)
