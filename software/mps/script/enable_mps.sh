#!/bin/bash
#export CUDA_VISIBLE_DEVICES="1"
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
