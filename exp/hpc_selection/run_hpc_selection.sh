#!/bin/bash

set -u

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

GPG=$ROOT_DIR/gnupg-1.4.13/g10/gpg

OUTPUT_FOLDER=$EXP_ROOT_DIR/hpc_selection/results
mkdir -p $OUTPUT_FOLDER
rm -rf $EXP_ROOT_DIR/hpc_selection/results/*

ARCHIVE_ROOT_FOLDER=$EXP_ROOT_DIR/hpc_selection/archive
CURRENT_TIME_SUFFIX=$(date +'%Y%m%d_%H%M%S')
ARCHIVE_FOLDER=$ARCHIVE_ROOT_FOLDER/$CURRENT_TIME_SUFFIX
mkdir -p $ARCHIVE_FOLDER

INTERVAL_US=1000
DATA_COLLECTION_TIME_S=10

SPs=('sensitive5')
SPcores=('0x80')
SPIDs=('')

clean_env () {
    sleep 1
    echo "Killing processes quickhpc, sensitive[1-9], sim_flush, sim_l3prime"
    ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sim_flush" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sim_l3prime" | awk '{print $2;}' | xargs -r kill
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

clean_env

for HPC_SEL in L3_TCM L3_TCA #BR_CN BR_INS BR_MSP BR_NTK BR_PRC FP_INS L1_DCM L1_ICM L1_LDM L1_STM L1_TCM L2_TCA L2_TCM L3_TCA L3_TCM LD_INS SR_INS STL_ICY TLB_DM TLB_IM TOT_CYC TOT_INS 
do
    for INTERFERE in none1 none2 sim_flush sim_l3prime
    do

        RUN_SAVE_DIR=$OUTPUT_FOLDER/
        mkdir -p $RUN_SAVE_DIR

        spawn_sensitive_programs
        if [[ "$INTERFERE" == "sim_flush" ]]
        then
            echo "Set sim_flush Core 0x8000"
            taskset 0x8000 ./sim_flush &
            sleep 10
        else
            if [[ "$INTERFERE" == "sim_l3prime" ]]
            then
                echo "Set sim_l3prime Core 0x8000"
                taskset 0x8000 ./sim_l3prime &
                sleep 10
            fi
        fi

        for i in ${!SPs[@]}
        do
            HPC_SUFFIX=${SPs[i]}_"$HPC_SEL"_"$INTERFERE"
            taskset 0x10 $quickhpc -c ./hpc_config/hpc_config_$HPC_SEL -a ${SPIDs[i]} -i $INTERVAL_US > $RUN_SAVE_DIR/$HPC_SUFFIX &
        done

        sleep $DATA_COLLECTION_TIME_S

        clean_env
    done
done

SCRIPT_NAME=$(basename -- "$0")
cp $SCRIPT_NAME $ARCHIVE_FOLDER/$SCRIPT_NAME
cp -r $OUTPUT_FOLDER $ARCHIVE_FOLDER/
