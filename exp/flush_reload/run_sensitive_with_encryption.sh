#!/bin/bash

set -u

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/flush_reload/results
mkdir -p $OUTPUT_FOLDER
rm -f $EXP_ROOT_DIR/flush_reload/results/*

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg
SENSITIVE_PROGRAM=sensitive5
SPY_PROGRAM=./spy
INTERVAL_US=100000
DATA_COLLECTION_TIME_S=20

clean_env () {
    sleep 1
    echo "Killing processes quickhpc, sensitive[1-9], spy, gnupg"
    ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "encrypt_" | awk '{print $2;}' | xargs -r kill
    sleep 1
}

clean_env

for SPLIT in TRAINING TESTING
do
  for HPC_COLLECTION in L1 L23 INS
  do

    HPC_SUFFIX=${HPC_COLLECTION}_${SPLIT}

    status "Encryption running"
    ./encrypt_rsa.sh &

    status "Sensitive program running"
    taskset 0x8 ./$SENSITIVE_PROGRAM $GPG&
    SENSITIVE_PID=$!

    taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_sensiprog_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep $DATA_COLLECTION_TIME_S

    clean_env

    status "Encryption running"
    ./encrypt_rsa.sh &

    status "Spy running"
    taskset 0x2000 $SPY_PROGRAM $GPG &
    SPY_PID=$!

    status "Sensitive program running"
    taskset 0x8 ./$SENSITIVE_PROGRAM $GPG&
    SENSITIVE_PID=$!

    taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_sensiprog_abnormal_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep $DATA_COLLECTION_TIME_S
    clean_env

  done
done
