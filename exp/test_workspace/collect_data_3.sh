#!/bin/bash

set -u

source ./config.sh

# Cleanup environment when exit
trap clean_env EXIT

clean_env

for SPEC_BG in perlbench bzip2 gcc mcf milc namd gobmk soplex povray hmmer sjeng libquantum h264ref lbm omnetpp astar
do
    for HPC_COLLECTION in SELECTED
    do
        for SPLIT in TRAIN TEST
        do
            RUN_SAVE_DIR=$OUTPUT_FOLDER/3/none/$SPEC_BG/
            mkdir -p "$RUN_SAVE_DIR"

            clean_env

            status "Encryption running"
            encrypt_large_file

            status "SPEC running" "$SPEC_BG"
            spec_background "$SPEC_BG"

            spawn_sensitive_programs

            for i in ${!SPs[@]}
            do
                HPC_SUFFIX=${SPs[i]}_${HPC_COLLECTION}_${SPLIT}
                taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a ${SPIDs[i]} -i $INTERVAL_US > $RUN_SAVE_DIR/hpc_$HPC_SUFFIX &
            done

            sleep $DATA_COLLECTION_TIME_S
        done
    done
done
