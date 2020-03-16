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
INTERVAL_US=100000
DATA_COLLECTION_TIME_S=10

#SPY_PROGRAM=./spy_fr
#SPs=('sensitive1' 'sensitive4' 'sensitive5')
#SPcores=('0x20' '0x80' '0x200')
#SPIDs=('' '' '')

SPY_PROGRAM=./spy_l1pp
SPs=('sensitive1')
SPcores=('0x20')
SPIDs=('')

clean_env () {
    sleep 1
    echo "Killing processes quickhpc, sensitive[1-9], spy, gnupg"
    ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "encrypt_" | awk '{print $2;}' | xargs -r kill
    sleep 1
}

spawn_sensitive_programs (){
    for i in "${!SPs[@]}"
    do
        sleep 1
        echo "Spawn" ${SPs[i]} "on core " ${SPcores[i]}
        taskset ${SPcores[i]} ./${SPs[i]} $GPG &
        SPIDs[i]=$!
    done
    sleep 1
}

clean_env
spawn_sensitive_programs


for SPLIT in TRAINING TESTING
do
  for HPC_COLLECTION in L1 L23 INS
  do

    status "Encryption running"
    ./encrypt_rsa.sh &

    spawn_sensitive_programs
    for i in ${!SPs[@]}
    do
        HPC_SUFFIX=${SPs[i]}_${HPC_COLLECTION}_${SPLIT}
        taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a ${gSPIDs[i]} -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_$HPC_SUFFIX &
    done

    sleep $DATA_COLLECTION_TIME_S

    clean_env

    status "Encryption running"
    ./encrypt_rsa.sh &

    status "Spy running"
    if ["$SPY_PROGRAM" == *"l1pp"*]
    then
        echo "Set" $SPY_PROGRAM "Core 0x8"
        taskset 0x8 $SPY_PROGRAM $GPG &
    else
        echo "Set" $SPY_PROGRAM "Core 0x2000"
        taskset 0x2000 $SPY_PROGRAM $GPG &
    fi
    spawn_sensitive_programs
    for i in ${!SPs[@]}
    do
        HPC_SUFFIX=${SPs[i]}_${HPC_COLLECTION}_${SPLIT}_abnormal
        taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a ${SPIDs[i]} -i $INTERVAL_US > $OUTPUT_FOLDER/hpc_$HPC_SUFFIX &
    done

    sleep $DATA_COLLECTION_TIME_S
    clean_env

  done
done
