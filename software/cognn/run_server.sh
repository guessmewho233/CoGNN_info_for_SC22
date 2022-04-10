#!/bin/bash
worker=2
CUDA_VISIBLE_DEVICES=0 python main.py ../data/model/mix_model.txt ${worker}
