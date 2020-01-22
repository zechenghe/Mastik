#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/ff/results

mkdir -p $OUTPUT_FOLDER

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg

status "Experiment begins"
status "Encryption"

while true;
  do
    taskset 0x4 $GPG -r 'zechengh_key1' -d 'hello.txt.gpg' &
    VICTIM_PID=$!
#    $quickhpc -c hpc_config -a $VICTIM_PID -i 100
    sleep 1
  done
