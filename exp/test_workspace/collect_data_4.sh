#!/bin/bash

set -u

source ./config.sh

# Cleanup environment when exit
trap clean_env EXIT

clean_env

for CACHE_ATTACK in fr ff l3pp l1pp
do
    for SPEC_BG in perlbench bzip2 gcc mcf milc namd gobmk soplex povray hmmer sjeng libquantum h264ref lbm omnetpp astar
    do
        for HPC_COLLECTION in SELECTED
        do
            for SPLIT in TRAIN TEST
            do
                RUN_SAVE_DIR=$OUTPUT_FOLDER/4/$CACHE_ATTACK/$SPEC_BG/
                mkdir -p "$RUN_SAVE_DIR"

                clean_env

                status "Encryption running"
                encrypt_large_file

                status "SPEC running" "$SPEC_BG"
                spec_background "$SPEC_BG"

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

                spawn_sensitive_programs

                for i in ${!SPs[@]}
                do
                    HPC_SUFFIX=${SPs[i]}_${HPC_COLLECTION}_${SPLIT}
                    taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a ${SPIDs[i]} -i $INTERVAL_US > $RUN_SAVE_DIR/4_hpc_$HPC_SUFFIX &
                done

                sleep $DATA_COLLECTION_TIME_S
            done
        done
    done
done

SCRIPT_NAME=$(basename -- "$0")
cp $SCRIPT_NAME $ARCHIVE_FOLDER/$SCRIPT_NAME
cp -r $OUTPUT_FOLDER $ARCHIVE_FOLDER/
