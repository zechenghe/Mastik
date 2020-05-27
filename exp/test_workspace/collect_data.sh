#!/bin/bash

source ./config.sh

mkdir -p $OUTPUT_FOLDER
rm -rf $EXP_ROOT_DIR/test_workspace/results/*
mkdir -p $ARCHIVE_FOLDER

# 0-a. Encryption
# 0-b. Encryption, attack
# 1. Sensitive prog, encryption
# 2. Sensitive prog, encryption, attack
# 3. Sensitive prog, encryption, backgournd(benchmark)
# 4. Sensitive prog, encryption, backgournd(benchmark), attack

./collect_data_0a.sh
./collect_data_0b.sh
./collect_data_1.sh
./collect_data_2.sh
./collect_data_3.sh
./collect_data_4.sh

SCRIPT_NAME=$(basename -- "$0")
cp $SCRIPT_NAME $ARCHIVE_FOLDER/$SCRIPT_NAME
cp ./collect_data_0a.sh $ARCHIVE_FOLDER/collect_data_0a.sh
cp ./collect_data_0b.sh $ARCHIVE_FOLDER/collect_data_0b.sh
cp ./collect_data_1.sh $ARCHIVE_FOLDER/collect_data_1.sh
cp ./collect_data_2.sh $ARCHIVE_FOLDER/collect_data_2.sh
cp ./collect_data_3.sh $ARCHIVE_FOLDER/collect_data_3.sh
cp ./collect_data_4.sh $ARCHIVE_FOLDER/collect_data_4.sh
cp -r $OUTPUT_FOLDER $ARCHIVE_FOLDER/
