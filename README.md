# Oort

This repository contains scripts and instructions for running our reproduction work based on "[Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/conference/osdi21/presentation/lai)".

# Overview

* [Getting Started](#getting-started)
* [Run Experiments and Validate Results](#run-experiments-and-validate-results)
* [Repo Structure](#repo-structure)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

# Getting Started

Run the following commands to install Oort.

```
git clone https://github.com/SymbioticLab/Oort
cd Oort
source install.sh
```

The following packages will be installed:

* Anaconda Package Manager
* CUDA 10.2

# Repo Structure

```
Repo Root
|---- selection        # Oort code base.
|---- federated
    |---- dataloader
    |---- aggregator and executor
|---- benchmark
    |---- configs
```

# Download Datasets

The datasets are available to download using the following commands:

```
git clone https://github.com/luowei2018/Oort-0.5.git
cd Oort/benchmark/dataset
bash download.sh download [dataset_name]
```

# Run Experiments
First, create a configuration file and put it to ```Oort-0.5/benchmark/configs```.
Then, run the experiment with ```federated driver start benchmark/configs/[config.yml]``` under ```Oort-0.5/``` directory.

# Plot Figures

Use ```ploort.py``` to create graphs regarding performance evaluations.

# Citation
```bibtex
@inproceedings{Oort-osdi21,
  title={Efficient Federated Learning via Guided Participant Selection},
  author={Fan Lai and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year={2021}
}
```
