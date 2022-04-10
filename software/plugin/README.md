# Plugin

This folder contains modifications to the PyTorch code (PyTorch v1.8.1). The modified PyTorch is for `CoGNN` or `PipeSwitch` server.

# PyTorch files

- memory.py: $PYTORCH\_PATH/torch/cuda/memory.py

- Module.cpp: $PYTORCH\_PATH/torch/csrc/cuda/Module.cpp

- CUDACachingAllocator.cpp: $PYTORCH\_PATH/c10/cuda/CUDACachingAllocator.cpp

- CUDACachingAllocator.h: $PYTORCH\_PATH/c10/cuda/CUDACachingAllocator.h

# Implementation

All the modified functions are labeled with a comment of "CoGNN" around them.

The code is implemented for NVIDIA V100, which have 32 GB GPU memory. Thus, some related parameters for memory management are directly written in the code. If you use GPUs with different GPU memory size, you need to change these parameters.
