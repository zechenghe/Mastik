#!/bin/bash

# Test whether HPCs of two different processes can be read at same time.

set -u

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/test_workspace/results
mkdir -p $OUTPUT_FOLDER
rm -f $EXP_ROOT_DIR/test_workspace/results/*

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg
SENSITIVE_PROGRAM_L1=sensitive1
SENSITIVE_PROGRAM_L3=sensitive5

SPY_PROGRAM_L1PP=./spy_l1pp
SPY_PROGRAM_L3PP=./spy_l3pp
SPY_PROGRAM_FF=./spy_ff
SPY_PROGRAM_FR=./spy_fr

ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill

INTERVAL_US=100000
DATA_COLLECTION_TIME_S=20


taskset 0x8 ./$SENSITIVE_PROGRAM_L1 &
SENSITIVE_PROGRAM_L1_PID=$!

sleep 1

taskset 0x8 ./$SENSITIVE_PROGRAM_L3 $GPG&
SENSITIVE_PROGRAM_L3_PID=$!

sleep 1

taskset 0x20 $quickhpc -c hpc_config_L1 -a $SENSITIVE_PROGRAM_L1_PID -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_sensiprog_L1 &
QUICKHPC_L1_PID=$!
taskset 0x20 $quickhpc -c hpc_config_L23 -a $SENSITIVE_PROGRAM_L3_PID -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_sensiprog_L23 &
QUICKHPC_L23_PID=$!
taskset 0x40 $quickhpc -c hpc_config_MISC -a $SENSITIVE_PROGRAM_L3_PID -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_sensiprog_MISC &
QUICKHPC_MISC_PID=$!

sleep 10

ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill
