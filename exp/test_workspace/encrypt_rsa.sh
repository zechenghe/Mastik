#!/bin/bash

set -u

source ./config.sh

while true
  do
    taskset 0x8 $GPG --batch -r zechengh_key1 -o /dev/null -e 'hello'
    sleep 0.01
  done
