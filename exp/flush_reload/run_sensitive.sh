#!/bin/bash

set -euxo pipefail

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/flush_reload/results
mkdir -p $OUTPUT_FOLDER
rm -f $EXP_ROOT_DIR/flush_reload/results/*

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg
SENSITIVE_PROGRAM=sensitive4
INTERVAL_US=10000

for SPLIT in TRAINING TESTING
do
  for HPC_COLLECTION in L1 L23 INS
  do
    HPC_SUFFIX=${HPC_COLLECTION}_${SPLIT}
    status "Sensitive program running"
    taskset 0x8 ./$SENSITIVE_PROGRAM &
    SENSITIVE_PID=$!

    taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_sensiprog_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep 100
    kill $QUICKHPC_PID
    kill $SENSITIVE_PID

    sleep 2

    status "Spy running"
    taskset 0x2000 ./spy 100000000 &
    SPY_PID=$!

    status "Sensitive program running"
    taskset 0x8 ./$SENSITIVE_PROGRAM &
    SENSITIVE_PID=$!

    taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_sensiprog_abnormal_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep 100
    kill $QUICKHPC_PID
    kill $SENSITIVE_PID
    kill $SPY_PID

    sleep 2

  done
done
