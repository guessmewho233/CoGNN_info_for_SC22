#!/bin/bash
network=("GCN" "GraphSAGE" "GAT" "GIN" "mix")
for i in `seq 0 9`;
do
    while true
    do
        process=`ps aux | grep mps_main | grep -v grep`
        if [ "$process" == "" ]; then
            taskId=`expr $i \* 2`    
            CUDA_VISIBLE_DEVICES=0 python mps_main.py ../data/model/${network[4]}_model.txt ${taskId} &
            taskId=`expr ${taskId} + 1`
            CUDA_VISIBLE_DEVICES=0 python mps_main.py ../data/model/${network[4]}_model.txt ${taskId}
            break
        fi
    done
done
