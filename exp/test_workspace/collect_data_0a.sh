#!/bin/bash

set -u

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/test_workspace/results
mkdir -p $OUTPUT_FOLDER
rm -rf $EXP_ROOT_DIR/test_workspace/results/*

ARCHIVE_ROOT_FOLDER=$EXP_ROOT_DIR/test_workspace/archive
CURRENT_TIME_SUFFIX=$(date +'%Y%m%d_%H%M%S')
ARCHIVE_FOLDER=$ARCHIVE_ROOT_FOLDER/$CURRENT_TIME_SUFFIX
mkdir -p $ARCHIVE_FOLDER

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg
INTERVAL_US=1000
DATA_COLLECTION_TIME_S=10

#SPY_PROGRAM=./spy_fr
#SPs=('sensitive1' 'sensitive4' 'sensitive5')
#SPcores=('0x8' '0x80' '0x80')
#SPIDs=('' '' '')

#SPY_PROGRAM=./spy_ff
#SPs=('sensitive5')
#SPcores=('0x80')
#SPIDs=('')

SPs=('sensitive1' 'sensitive5')
SPcores=('0x8' '0x80')
SPIDs=('' '')

clean_env () {
    echo "Killing processes quickhpc, sensitive[1-9], spy, gnupg"
    ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "zechengh_key1" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "encrypt_" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "runspec" | awk '{print $2;}' | xargs -r kill
    mkdir -p $OUTPUT_FOLDER
}

spec_background(){
    taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 "$1" &
    echo taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 "$1"
    sleep 1
}

# Cleanup environment when exit
trap clean_env EXIT

spawn_sensitive_programs (){
    for i in "${!SPs[@]}"
    do
        sleep 1
        echo "Spawn" ${SPs[i]} "on core " ${SPcores[i]}
        taskset ${SPcores[i]} ./${SPs[i]} $GPG &
        SPIDs[i]=$!
    done
    sleep 5
}

encrypt_large_file (){
    taskset 0x8 $GPG -r zechengh_key1 -o /dev/null -e ~/cuda_10.1.105_418.39_linux.run &
    ENC_PID=$!
    sleep 1
}

clean_env

mkdir -p $OUTPUT_FOLDER/0a
for HPC_COLLECTION in SELECTED
do
    for SPLIT in TRAIN TEST
    do
        clean_env
        status "Encryption running"
        encrypt_large_file

        HPC_SUFFIX=enc_${HPC_COLLECTION}_${SPLIT}
        taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $ENC_PID -i $INTERVAL_US > $OUTPUT_FOLDER/0a/0a_hpc_$HPC_SUFFIX &
        sleep $DATA_COLLECTION_TIME_S
    done
done

SCRIPT_NAME=$(basename -- "$0")
cp $SCRIPT_NAME $ARCHIVE_FOLDER/$SCRIPT_NAME
cp -r $OUTPUT_FOLDER $ARCHIVE_FOLDER/
