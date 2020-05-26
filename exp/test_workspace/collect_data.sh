#!/bin/bash

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
