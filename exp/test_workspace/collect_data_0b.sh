#!/bin/bash

set -u

source ./config.sh

# Cleanup environment when exit
trap clean_env EXIT

clean_env

for CACHE_ATTACK in fr ff l3pp l1pp
do
    for HPC_COLLECTION in SELECTED
    do
        for SPLIT in TRAIN TEST
        do
            RUN_SAVE_DIR=$OUTPUT_FOLDER/0b/$CACHE_ATTACK/none/
            mkdir -p "$RUN_SAVE_DIR"

            clean_env
            status "Encryption running"
            encrypt_large_file

            status "Spy running"
            SPY_PROGRAM=./spy_$CACHE_ATTACK
            if [[ "$SPY_PROGRAM" == *"l1pp"* ]]
            then
                echo "Set" $SPY_PROGRAM "Core 0x8000"
                taskset 0x8000 $SPY_PROGRAM 1000000000 &
            else
                echo "Set" $SPY_PROGRAM "Core 0x2000"
                if [[ "$SPY_PROGRAM" == *"l3pp"* ]]
                then
                    taskset 0x2000 $SPY_PROGRAM 1000000000 &
                else
                    taskset 0x2000 $SPY_PROGRAM $GPG &
                fi
            fi

            sleep 5

            HPC_SUFFIX=enc_${HPC_COLLECTION}_${SPLIT}
            taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a $ENC_PID -i $INTERVAL_US > $RUN_SAVE_DIR/hpc_$HPC_SUFFIX &
            sleep $DATA_COLLECTION_TIME_S
        done
    done
done
