# T-GLean

This is the source code that accompanies the paper "Goal-directed Planning and Goal Understanding by Active Inference: Evaluation Through Simulated and Physical Robot Experiments"
This implementation of T-GLean is based on a pre-release version of LibPvrnn, an implementation of the Predictive-coding-inspired Variational Bayes Recurrent Neural Network (PV-RNN), written in C++ and Python.

This document will show how to get started in reproducing the results shown in Experiment 1 of the aforementioned paper. It is not intended as a general manual on how to use LibPvrnn. As this codebase has forked off the main development branch of LibPvrnn some time ago, we cannot guarantee forward compatibility with later versions of LibPvrnn.

LibPvrnn is developed by Wataru Ohata, Takazumi Matsumoto and Fabien Benureau. LibPvrnn is licensed under GPLv3.

## Requirements
LibPvrnn has been confirmed to work on most modern operating systems:
- Linux (tested on Ubuntu 20.04)
- macOS (there may be some issues with Macs running on Apple Silicon)
- Windows 10 (using WSL/Ubuntu)

Prerequisites to build LibPvrnn:
- a GCC-compatible C++ compiler that supports C++17 (tested on GCC 9.3, 10.3, and 11.1)
- CMake 3.11+
- Eigen 3.3+ (tested on 3.3.7 and 3.4.0)
- Zlib
- Python 3.7+ (tested on 3.8.10)

The following section will show how to build LibPvrnn and run the T-GLean planner.

## Using T-GLean
This repository contains the configuration files (in the configs/glean subdirectory), datasets (in the datasets subdirectory) and pre-trained networks (in the results subdirectory) used in Experiment 1 of the aforementioned paper.

### Build
After ensuring all the prerequisites above are installed, build LibPvrnn by executing the following:
```bash
mkdir build
cd build
cmake ..
make
```

### Running the planner
The planning scripts are contained within the planner/python subdirectory.

Note: if you want to execute the scripts from a different directory or have changed the directory structure, please set the PVRNN_SAVE_DIR environment variable to point to the directory that contains the results subdirectory.

### Training new networks
Aside from the included pre-trained networks, it is possible to train a network from scratch. A configuration file and dataset in NPY format is required.

To train a network in either C++ or Python:
```bash
bin/gtrain ../configs/my_config.cfg
python ../planner/python/gtrain.py ../configs/my_config.cfg
```

To test generation (prior and posterior) in either C++ or Python:
```bash
bin/gtest ../configs/my_config.cfg
python ../planner/python/gtest.py ../configs/my_config.cfg
```

A full explanation of the configuration, implementation or use of LibPvrnn is beyond the scope of this document. Please contact the authors for more details on LibPvrnn.
