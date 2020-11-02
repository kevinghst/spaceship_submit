# Spaceship Implementation

## Requirements
In addition to those in requirements.txt, following dependencies are needed

    cudatoolkit 10.2
    cuda 10.1
    gcc 6.3.0


## Training Instructions

First, compile the CUDA extension. (not needed for evaluation)

    cd cuda_op
    python setup.py install

Then, run train.py

    cd ..
    python train.py

## Reference
The code for GIoU loss is copied from https://github.com/lilanxiao/Rotated_IoU