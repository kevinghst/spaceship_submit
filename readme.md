# Spaceship Implementation
This repository contains implementation/solution to the following Computer Vision assignment:

**Problem:**
The goal is to detect spaceships which have been fitted with a cloaking device that makes them less visible. You are expected to use a deep learning model to complete this task. The model will take a single channel image as input and detects the spaceship (if it exists). Not all image will contain a spaceship, but they will contain no more than 1. For any spaceship, the model should predict their bounding box and heading. This can be described using five parameters:

* X and Y position (centre of the bounding box)
* Yaw (direction of heading)
* Width (size tangential to the direction of yaw)
* Height (size along the direct of yaw)

A sample spaceship image is given in `example.png`.

The metric for the model is AP at an IOU threshold of 0.7, for at least 1000 random samples, with the default generation parameters (see `main.py`).

**Evaluation Criteria:**
* Model metric, score as high as you can while being under 2 million trainable parameters. *The average submitted AP for this assignment is 0.7*.
* Model architecture
* Loss function
* Code readability and maintainability, please follow general python conventions

**Deliverables**
1. Final score reported in `final_score.txt`
1. A summary of the model architecture in `architecture_summary.txt`
1. A `train.py` script that allows the same model to be reproduced
1. The final model weights in `model.pth.tar`
1. A `requirements.txt` file that includes all python dependencies and their versions
1. A `main.py` file that reproduces the reported score

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
`plot_examples` and parts of `helpers.py` (code for generating spaceship images) were given by the assignment.

The code for GIoU loss is copied from https://github.com/lilanxiao/Rotated_IoU, and is contained in directory `cuda_op` and files `box_intersection_2d.py`, `min_enclosing_box.py`, `oriented_iou_loss.py`, `utiles.py`.

`main.py`, `network.py`, `train.py`, and rest of `helpers.py` are implemented from scratch.