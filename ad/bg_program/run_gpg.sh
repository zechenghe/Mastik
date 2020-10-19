#!/bin/bash

set -u

while true
  do
    gpg --batch -r zechengh_key1 -o /dev/null -e /home/zechengh/sample.txt
  done
