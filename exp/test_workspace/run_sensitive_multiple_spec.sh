#!/bin/bash

set -u

quickhpc="/home/zechengh/quickhpc/quickhpc"

ROOT_DIR="/home/zechengh/Mastik"
EXP_ROOT_DIR=$ROOT_DIR/exp
source $EXP_ROOT_DIR/exp_funcs.sh

OUTPUT_FOLDER=$EXP_ROOT_DIR/test_workspace/results
mkdir -p $OUTPUT_FOLDER
rm -rf $EXP_ROOT_DIR/test_workspace/results/*

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

SPY_PROGRAM=./spy_fr
SPs=('sensitive1' 'sensitive5')
SPcores=('0x8' '0x80')
SPIDs=('' '')

clean_env () {
    sleep 1
    echo "Killing processes quickhpc, sensitive[1-9], spy, gnupg"
    ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "zechengh_key1" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "encrypt_" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "runspec" | awk '{print $2;}' | xargs -r kill
    sleep 1
}

spec_background(){
    taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 "$1" &
    echo taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 "$1"
    sleep 1
}

multi_spec_background(){
    taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 perlbench &
    echo taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 perlbench

    sleep 1

    taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 libquantum &
    echo taskset 0x20 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 libquantum

    sleep 1

    #taskset 0x200 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 hmmer &
    #echo taskset 0x200 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 hmmer

    #sleep 1

    #taskset 0x200 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 astar &
    #echo taskset 0x200 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 astar

    #sleep 1

    #taskset 0x8 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 lbm &
    #echo taskset 0x8 runspec --config=test.cfg --size=train --noreportable --tune=base --iterations=1 lbm

    #sleep 1
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

#for SPEC_BG in perlbench none bzip2 gcc mcf milc namd gobmk soplex povray hmmer sjeng libquantum h264ref lbm omnetpp astar
#do

for SPEC_BG in none multiple
do
    mkdir -p $OUTPUT_FOLDER/$SPEC_BG
    for SPLIT in TRAINING TESTING
    do
        for HPC_COLLECTION in OLD OLD_L3
        do
            status "Encryption running"
            encrypt_large_file

            if [[ "$SPEC_BG" != "none" ]]
            then
                multi_spec_background
            fi

            spawn_sensitive_programs

            for i in ${!SPs[@]}
            do
                HPC_SUFFIX=${SPs[i]}_${HPC_COLLECTION}_${SPLIT}
                taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a ${SPIDs[i]} -i $INTERVAL_US > $OUTPUT_FOLDER/$SPEC_BG/hpc_$HPC_SUFFIX &
            done

            sleep $DATA_COLLECTION_TIME_S

            clean_env

            status "Encryption running"
            encrypt_large_file

            status "Spy running"
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

            if [[ "$SPEC_BG" != "none" ]]
            then
                multi_spec_background
            fi

            spawn_sensitive_programs

            for i in ${!SPs[@]}
            do
                HPC_SUFFIX=${SPs[i]}_${HPC_COLLECTION}_${SPLIT}_abnormal
                taskset 0x10 $quickhpc -c hpc_config_$HPC_COLLECTION -a ${SPIDs[i]} -i $INTERVAL_US > $OUTPUT_FOLDER/$SPEC_BG/hpc_$HPC_SUFFIX &
            done

            sleep $DATA_COLLECTION_TIME_S

            clean_env
    done
  done
done
