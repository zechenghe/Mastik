#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/l1pp/results
mkdir -p $OUTPUT_FOLDER
rm -f $EXP_ROOT_DIR/l1pp/results/*

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg

clean_env () {
   echo "Killing processes quickhpc, sensitive[1-9], spy, gnupg"
   ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
   ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
   ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill
   ps -ef | grep "encrypt_" | awk '{print $2;}' | xargs -r kill
}


for SPLIT in TRAINING TESTING
do
  for HPC_COLLECTION in L1 L23 MISC
  do
    HPC_SUFFIX=${HPC_COLLECTION}_${SPLIT}
    status "Sensitive program running"
    taskset 0x8 ./sensitive1 &

    status "Encryption running"
    ./encrypt_rsa.sh &

    $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i 1000 > $OUTPUT_FOLDER/hpc_sensiprog_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep 10

    clean_env


    status "Spy running"
    taskset 0x8000 ./spy 1000000000 &

    status "Sensitive program running"
    taskset 0x8 ./sensitive1 &

    status "Encryption running"
    ./encrypt_rsa.sh &

    $quickhpc -c hpc_config_$HPC_COLLECTION -a $SENSITIVE_PID -i 100 > $OUTPUT_FOLDER/hpc_sensiprog_abnormal_$HPC_SUFFIX &
    QUICKHPC_PID=$!

    sleep 10

    clean_env

  done
done
