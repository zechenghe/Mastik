#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=EXP_ROOT_DIR/l1pp/results

mkdir -p $OUTPUT_FOLDER

status "Run victim"
taskset 0x8000 ./victim &
VICTIM_PID=$!
echo $VICTIM_PID
