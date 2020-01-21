#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/l1pp/results

mkdir -p $OUTPUT_FOLDER

status "Attacker running"
taskset 0x8000 ./spy 1000000000 &
SPY_PID=$!
echo $SPY_PID

sleep 3

kill $SPY_PID

# $quickhpc -c hpc_config -a $SPY_PID -i 1000 > $OUTPUT_FOLDER/hpc_attacker

echo "Attacker ends"
