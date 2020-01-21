#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/l1pp/results

mkdir -p $OUTPUT_FOLDER

status "Run victim"
taskset 0x8 ./victim &
VICTIM_PID=$!
echo $VICTIM_PID

$quickhpc -c hpc_config -a $VICTIM_PID -i 1000 #> $OUTPUT_FOLDER/hpc_victim &

sleep 10

kill $VICTIM_PID
echo "Experiment ends"
