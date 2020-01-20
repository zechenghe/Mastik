#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/flush_reload/results

[ -d $OUTPUT_FOLDER ] && rm -r $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg

status "Attacker running"
taskset 0x2 ./spy $GPG &
SPY_PID=$!
echo $SPY_PID
