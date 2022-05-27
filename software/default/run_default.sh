#!/bin/bash

network=("GCN" "GraphSAGE" "GAT" "GIN" "mix")
for i in `seq 0 19`;
do
    CUDA_VISIBLE_DEVICES=0 python default_main.py ../data/model/${network[4]}_model.txt ${i}
done
