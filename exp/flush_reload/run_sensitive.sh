#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/flush_reload/results

mkdir -p $OUTPUT_FOLDER

status "Attacker running"
taskset 0x40000 ./sensitive1 &
SENSITIVE_PID=$!
echo $SENSITIVE_PID

$quickhpc -c hpc_config -a $SENSITIVE_PID -i 100 > $OUTPUT_FOLDER/hpc_sensiprog
