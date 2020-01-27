#!/bin/bash

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/l1pp/results
mkdir -p $OUTPUT_FOLDER

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg

for HPC_SUFFIX in L1 L23 MISC
do

  status "Sensitive program running"
  taskset 0x8000 ./sensitive1 &
  SENSITIVE_PID=$!
  echo $SENSITIVE_PID

  $quickhpc -c hpc_config_$HPC_SUFFIX -a $SENSITIVE_PID -i 1000 > $OUTPUT_FOLDER/hpc_sensiprog_$HPC_SUFFIX &
  QUICKHPC_PID=$!

  sleep 5
  kill $QUICKHPC_PID
  kill $SENSITIVE_PID


  status "Spy running"
  taskset 0x8000 ./spy 1000000000 &
  SPY_PID=$!

  status "Sensitive program running"
  taskset 0x8 ./sensitive1 &
  SENSITIVE_PID=$!
  echo $SENSITIVE_PID

  $quickhpc -c hpc_config_$HPC_SUFFIX -a $SENSITIVE_PID -i 1000 > $OUTPUT_FOLDER/hpc_sensiprog_abnormal_$HPC_SUFFIX &
  QUICKHPC_PID=$!

  sleep 5
  kill $QUICKHPC_PID
  kill $SENSITIVE_PID
  kill $SPY_PID

done
