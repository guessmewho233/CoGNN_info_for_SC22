# MPS

This folder contains `MPS` code and scripts, also modification to the PyTorch code (PyTorch v1.8.1).

## Files

- PyTorch file
    - CUDACachingAllocator.cpp: $PYTORCH\_PATH/c10/cuda/CUDACachingAllocator.cpp

- script
    - enable\_mps.sh: set GPU to EXCLUSIVE mode and start MPS service.
    - stop\_mps.sh: quit MPS service and set GPU to DEFAULT mode.

- mps\_main.py: execute one training task.

- run\_mps.sh: execute pairwise training tasks in order. 

## Usage

```
./run_mps.sh
```
