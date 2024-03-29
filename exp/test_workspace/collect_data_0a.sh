#!/bin/bash

set -u

source ./config.sh

# Cleanup environment when exit
trap clean_env EXIT

clean_env

for HPC_COLLECTION in SELECTED
do
    for SPLIT in TRAIN TEST
    do
        RUN_SAVE_DIR=$OUTPUT_FOLDER/0a/none/none/
        mkdir -p "$RUN_SAVE_DIR"

        clean_env
        status "Encryption running"
        encrypt_large_file

        HPC_SUFFIX=enc_${HPC_COLLECTION}_${SPLIT}
        taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $ENC_PID -i $INTERVAL_US > $RUN_SAVE_DIR/hpc_$HPC_SUFFIX &
        sleep $DATA_COLLECTION_TIME_S
    done
done
