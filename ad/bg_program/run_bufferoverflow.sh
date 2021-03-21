#!/bin/bash

set -u

while true
  do
    taskset 0x8 env - /home/zechengh/Mastik/ad/attack/bufferoverflow/victim.o
  done
