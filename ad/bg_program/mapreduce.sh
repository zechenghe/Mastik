#!/bin/bash

set -u

$HADOOP_HOME/bin/hadoop fs -rm -r /user/zechengh/terasort/out
while true
  do
    $HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.0.jar terasort /user/zechengh/terasort/randinputlarge/ /user/zechengh/terasort/out;
    $HADOOP_HOME/bin/hadoop fs -rm -r /user/zechengh/terasort/out
  done
