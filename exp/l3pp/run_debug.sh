#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/l3pp/results
mkdir -p $OUTPUT_FOLDER

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg
SENSITIVE_PROGRAM=sensitive2

for SPLIT in TRAINING TESTING
do
  for HPC_COLLECTION in DEBUG L23
  do
    HPC_SUFFIX=${HPC_COLLECTION}_${SPLIT}
    status "Sensitive program running"
    taskset 0x8 ./$SENSITIVE_PROGRAM &
    SENSITIVE_PID=$!

    taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i 1000 > $OUTPUT_FOLDER/hpc_sensiprog_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep 10
    kill $QUICKHPC_PID
    kill $SENSITIVE_PID

    sleep 1

    status "Spy running"
    taskset 0x2000 ./spy 1000000 &
    SPY_PID=$!

    status "Sensitive program running"
    taskset 0x8 ./$SENSITIVE_PROGRAM &
    SENSITIVE_PID=$!

    taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i 1000 > $OUTPUT_FOLDER/hpc_sensiprog_abnormal_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep 10
    kill $QUICKHPC_PID
    kill $SENSITIVE_PID
    kill $SPY_PID

    sleep 1

  done
done
