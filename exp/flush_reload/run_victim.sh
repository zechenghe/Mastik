#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/flush_reload/results

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

#status "Attacker starts"
#./attacker $SHARED_MEM > $OUTPUT_FOLDER/0

#status "Victim starts"
#./victim $SHARED_MEM &
#VICTIM_PID=$!

#./attacker $SHARED_MEM > $OUTPUT_FOLDER/1

#kill "$VICTIM_PID"
#status "Experiment ends"
