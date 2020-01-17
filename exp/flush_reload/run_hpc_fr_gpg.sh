#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=EXP_ROOT_DIR/flush_reload/results

[ -d $OUTPUT_FOLDER ] && rm -r $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg

status "Experiment begins"
status "Encryption"

$GPG -r 'zechengh_key1' -d 'hello.txt.gpg' &
VICTIM_PID=$!

$quickhpc -c hpc_config -a $VICTIM_PID -i 200

sleep 2

status "Attacker running"
./spy $GPG &
SPY_PID=$!

sleep 2

$GPG -r 'zechengh_key1' -d 'hello.txt.gpg' &
VICTIM_PID=$!

$quickhpc -c hpc_config -a $VICTIM_PID -i 200

kill $SPY_ID

#status "Attacker starts"
#./attacker $SHARED_MEM > $OUTPUT_FOLDER/0

#status "Victim starts"
#./victim $SHARED_MEM &
#VICTIM_PID=$!

#./attacker $SHARED_MEM > $OUTPUT_FOLDER/1

#kill "$VICTIM_PID"
#status "Experiment ends"
