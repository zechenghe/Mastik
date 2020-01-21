#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/l1pp/results

[ -d $OUTPUT_FOLDER ] && rm -r $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER

status "Attacker running"
taskset 0x8 ./spy 1000000000 &
SPY_PID=$!
echo $SPY_PID

$quickhpc -c hpc_config -a $SPY_PID -i 100000 > $OUTPUT_FOLDER/hpc

echo "Experiment end"
