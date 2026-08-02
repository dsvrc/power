## Installation

This folder includes the **topology optimization (discrete)** tasks with exponentially large action spaces and takes inspiration from [CleanRL](https://github.com/vwxyzjn/cleanrl) clean design principles to provide a simple codebases with flexible MARL configurations and implementations of popular MARL baselines. The user must set up [wandb](https://wandb.ai/home) to run and log experiment results.  

In contrast to the redispatching and curtailment (continuous) case, the topology optimization (discrete) tasks require a multi-agent compatible version of Grid2Op (i.e., v11.0_dev). For this reason we keep the two versions of the tasks in separate code bases.

1. Download [Miniconda](https://docs.anaconda.com/free/miniconda/) for your system.  
2. Install Miniconda.  
3. Go to the Topology_Task main folder:  
    ```bash
    cd Topology_Task
    ```  
4. Set up the conda environment:  
    ```bash
    conda env create -f conda_env.yml
    ```  
5. Activate the conda environment:  
    ```bash
    conda activate marl2grid
    ```  
6. Install Grid2Op (in the correct multi-agent version) from the correct Github repository and branch here: 
    https://github.com/Grid2op/grid2op/tree/dev_multiagent
7. Install Topology_Task:  
    ```bash
    pip install .
    ```  

---

## Usage

Run the `main.py` script with the desired parameters and task configuration.  

- Algorithm-specific parameters are defined under `alg/<algorithm>/config.py`.  
- Multi-agent setups follow the agent–substation partitions described in the paper and supplementary material.  

For logging and tracking experiments, check the argument parser in `main.py` and the integration with [Weights & Biases](https://wandb.ai/home).  

---

## Experiments

To run training on a predefined task (remember to set up the correct entity and project for wandb in the `main.py` script):

```bash
python main.py --env-id bus14 --alg MAPPO
```

Available arguments include which observability regime to use, whether to use the idle heuristic or not, constraint types, and more. Check `main.py`, `alg/<algorithm>/config.py`, `env/config.py` for the full configuration space.

It is possible to reproduce the paper results by running each algorithm with the hyperparameters specified in our supplementary material. Final runs were trained with the seeds 0, 1, 2, 3 and 4.

The `--checkpoint` flag allows to save checkpoints. Checkpoints can be heavy because they save everything, including the memory buffer.

---

## Evaluation

There is a deterministic evaluation module embedded in the code. Policies will be tested at training time at results will be logged on wandb.

```
