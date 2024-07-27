# DL 2 Project - Reproduction of E(n) Equivariant Simplicial Message Passing Network

This project is meant to showcase the effort in replication the paper of Eijkelboom et al about message passing along simplices. Our aim was to re-implement the proposed network architecture in the up and comming TopoX package suite. This suite is a newly developed set of packages that facilitate the pipeline of Topological Deep Learning, from lifting techniques to actual models of message passing.

## Setup
To setup from a clean environment we will reproduce the steps as mentioned in the `challenge-icml-2024` and an additional one.

1. Create the conda environment to be used
   ```bash
   conda create -n topox python=3.11.3
   conda activate topox
   ```
2. Install the required packages and the TopoX suite including the code reqquiremed from the `challenge-icml-2024` repo
   ```bash
   pip install -e '.[all]'
   ```
3. Install pytorch related libraries

      ```bash
      pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu115
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu115.html
      pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu115.html
      ```
4. Wandb for saving model information
    ```bash
    pip install wandb
    ```
## Experiments

There are two important directions the experiments follow: execution efficiency, performance of the network. We evaluate the performance of the forward pass and lifting procedure.
performance in the QM9 dataset.

To execute an experiment activate the `topox` environment created above and go into `src`.

The file `main.py` can be used to run the experiment, however `main_light.py` is an implementation using Pytorch Lighting that achieves better times some optional paramters are explained below.

+ `--lift_type <lift>`: the type of lifting procedure `alpha` or `rips`
+ `--batch_size <size>`: the size of each batch
+ `--dim <int>`: The maximum dimension of the complex
+ `--dis <float>`: The delta parameter in the VR Filtration (diameter of the growing ball)
+ `--target_name <name>`: the target molecular property to train/predict for
+ `--debug `: to run a smaller subset of the dataset for testing purpouses
+ `--pre_proc`: wether the invariances should be precomputed during the lift procedure or not (beware it's vary time consuming)
+ `--benchmark`: Runs the process using benchmarking distincts so that it logs to WandDB

