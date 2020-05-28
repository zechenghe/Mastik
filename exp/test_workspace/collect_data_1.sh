#!/bin/bash

set -u

source ./config.sh

# Cleanup environment when exit
trap clean_env EXIT

clean_env

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

mkdir -p $OUTPUT_FOLDER/1
for HPC_COLLECTION in SELECTED
do
    for SPLIT in TRAIN TEST
    do
        RUN_SAVE_DIR=$OUTPUT_FOLDER/none/none/1
        mkdir -p "$RUN_SAVE_DIR"

        clean_env
        status "Encryption running"
        encrypt_large_file

        spawn_sensitive_programs

        for i in ${!SPs[@]}
        do
            HPC_SUFFIX=${SPs[i]}_${HPC_COLLECTION}_${SPLIT}
            taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a ${SPIDs[i]} -i $INTERVAL_US > $RUN_SAVE_DIR/hpc_$HPC_SUFFIX &
        done

        sleep $DATA_COLLECTION_TIME_S
    done
done
