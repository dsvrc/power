
## Installation

In contrast to the topological (discrete) case, the curtailment and redispatching (continuous) task requires a fork
of our own of the `BenchMARL` python library.

1. Download [Miniconda](https://docs.anaconda.com/free/miniconda/) for your system.  
2. Install Miniconda.  
3. Go to the Redispatching_Curtailment_Task main folder:  
    ```bash
    cd Redispatching_Curtailment_Task
    ```  
4. Set up the conda environment:  
    ```bash
    conda env create -f conda_env.yml
    ```  
5. Activate the conda environment:  
    ```bash
    conda activate marl2grid
    ```  
6. Install BenchMARL from the repository. Go to the BenchMARL folder:
    ```bash
    cd ../BenchMARL
    ```
7. Install BenchMARL
    ```bash
    pip install . --constraint ../requirements.txt
    ```  

Note that `tensordict` 0.7.2 requires at least GCC 11. See this [issue](https://github.com/pytorch/tensordict/issues/1235).

## Usage

Run the `main.py` script with the desired parameters and task configuration.  

- Common parameters are defined in `configs/expes_config.yaml`.  
- Multi-agent setups follow the agent–substation partitions described in the paper and supplementary material.  

---

## Experiments

To run training on a predefined task :

```bash
python main.py --n_frames 1_000_000 --algo MAPPO --lr 3e-5 --MAPPO_n_episode 30
```

The hyperparameter values we obtained with grid search have been set as defaults in `main.py`.

It is possible to reproduce the paper results by running each algorithm with the hyperparameters specified in our supplementary material. Final runs were trained with the seeds 0, 1, 2, 3 and 4.

The `--save_experiment` option allows saving the final checkpoint. But note that a MASAC (off-policy algorithm) checkpoint can be heavy because `benchMARL` seems to also save the buffer. It does not apply to MAPPO, which is an on-policy algorithm.

To avoid having to save the models, we have added the `--evaluate_agents` option, which evaluates each agent as soon as its training is complete.

## Evaluation

Before starting an evaluation, you have to download the test set via this [reviewer link](https://osf.io/bgn4c/?view_only=238440aa5af645598c398b933703846a). Then, unzip and move the folder `l2rpn_idf_2023_test_new` to the data `grid2op` folder. You can identify it with the [`grid2op.get_current_local_dir`](https://grid2op.readthedocs.io/en/latest/makeenv.html#grid2op.MakeEnv.get_current_local_dir) method:

```python
import grid2op
print(f"Data about grid2op downloaded environments are stored in: "{grid2op.get_current_local_dir()}"")
```

You are now ready to evaluate an agent. Either use the `--evaluate_agents` option in the `main.py` script, or use the `evaluate.py` script, which evaluates a saved agent.

```bash
python evaluate.py --agent_name <agent_name> 
```

---

## Loggers

A list of logger names can be provided in the experiment config. Examples of available options are: `wandb`, `csv`, `mflow`, `tensorboard`. In `configs/expes_config.yaml`, replace `loggers: []` by `loggers: ["csv", "tensorboard"]`.

By default, we do not use loggers, as they can cause `FileNotFoundError` on Windows if the root path contains too many characters (on Windows, file paths are limited by default to 260 characters).