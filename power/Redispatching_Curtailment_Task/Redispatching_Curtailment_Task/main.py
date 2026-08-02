
import os

# BLAS threading has to be pinned before torch/numpy are imported. Every collector env is a
# forked process; without this each one spawns a full thread pool and they fight over the
# same cores, which is slower than running serially. setdefault, so an explicit export wins.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import yaml
import argparse
import gc
import torch
from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.environments import G2OpPowerGridTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import grid2op
from utils import ROOT_DIR, G2OP_ENV_DIR, IS_LINUX
from BMMAAgent import BMMAAgent
from evaluate import evaluate

def available_cpus():
    """Cores actually allocated to this process (respects the SLURM/cgroup cpuset)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def n_envs_arg(value):
    return value if value == "auto" else int(value)


def resolve_n_envs(n_envs, frames_per_batch):
    """Collection is split evenly across workers, so the worker count has to divide
    frames_per_batch. Round the request down to the nearest divisor rather than failing:
    asking for 32 on a 32-core node quietly gives 30, which is what you want anyway.
    Returns (resolved, requested)."""
    requested = available_cpus() if n_envs in (None, "auto") else n_envs
    requested = max(1, min(requested, frames_per_batch))
    resolved = max(d for d in range(1, requested + 1) if frames_per_batch % d == 0)
    return resolved, requested


def cli():
    parser = argparse.ArgumentParser(description="Train some agents.")
    parser.add_argument('--n_frames', type=int, default=6_000,
                        help="Total number of frames to collect for training. (default: 6_000)")
    parser.add_argument('--lr', type=float, default=3e-5,
                        help="Learning rate. (default: 3e-5)")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Gamma. (default: 0.99)")
    parser.add_argument('--frames_per_batch', type=int, default=6000,
                        help="Frames per batch. (default: 6000)")
    parser.add_argument('--MAPPO_n_episode', type=int, default=30,
                        help="Number of episodes when training with MAPPO. (default: 15)")
    parser.add_argument('--MASAC_n_optimizer_steps', type=int, default=1000,
                        help="""Number of times MASAC_train_batch_size will be sampled from 
                        the buffer and trained over when training with MASAC. (default: 1000)""")
    parser.add_argument('--MASAC_train_batch_size', type=int, default=256,
                        help="""Number of frames used for each optimizer's step when training 
                                with MASAC. (default: 128)""")
    parser.add_argument('--seeds', type=int, default=[0, 1, 2], nargs='+',
                        help="Random seeds (default: [0, 1, 2])")
    parser.add_argument('--alg', type=str, default="MAPPO", choices=["MAPPO", "MASAC"],
                        help="Whether to train with MAPPO or MASAC. (default: MAPPO)")
    parser.add_argument('--save_experiment', action='store_true', 
                        help="""Whether or not to save the experiment. Note that a MASAC checkpoint
                                can be heavy because the buffer is also saved in it. (default: False)""")
    parser.add_argument('--evaluate_agents', action='store_true',
                        help="Whether or not to evaluate the trained agents. (default: False)")
    parser.add_argument('--n_envs', type=n_envs_arg, default="auto",
                        help="""Number of CPUs to collect on, i.e. how many environments run in
                                parallel. Rounded down to the nearest divisor of frames_per_batch.
                                "auto" (default) uses every core allocated to the job.""")
    parser.add_argument('--n_train_threads', type=int, default=None,
                        help="""Torch intra-op threads for the optimizer loop. Only takes effect
                                once the collector workers exist, so they stay single-threaded
                                under OMP_NUM_THREADS=1. (default: leave unchanged)""")
    return parser.parse_args()

def train_algo(task, algorithm_config, model_config, critic_model_config, experiment_config, seed, evaluate_agent,
               n_train_threads=None):
        print("Creating experiment...")
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=experiment_config,
        )
        # The optimizer loop runs here in the parent, so it can use more threads than the
        # collector workers. Set this only after the collector (and its workers) exist.
        if n_train_threads is not None:
            torch.set_num_threads(n_train_threads)
        print("Starting training...")
        experiment.run()
        experiment.close()

        # Evaluation
        if evaluate_agent:
            print("Starting evaluation...")

            env = grid2op.make(os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023_test_new"))
            # Load the agent
            grid2op_agent = BMMAAgent(env.action_space, nn_kwargs={})
            grid2op_agent.load(experiment=experiment)
            print("Agent loaded.")

            # Evaluate the agent
            algo = algorithm_config.associated_class().__name__.upper()
            evaluate(grid2op_agent, agent_name=f"{algo}_{seed}", results_path_agents=os.path.join(ROOT_DIR, "agents_results"))

        return experiment


if __name__ == "__main__":

    if IS_LINUX:
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)

    args = cli()
    args_dict = vars(args)
    n_frames, lr, gamma, frames_per_batch, MAPPO_n_episode, MASAC_n_optimizer_steps, MASAC_train_batch_size, seeds, alg, save_experiment, evaluate_agents, n_envs, n_train_threads = args_dict.values()

    n_envs, n_cpus_requested = resolve_n_envs(n_envs, frames_per_batch)
    print(f"[parallelism] {available_cpus()} cores visible, {n_cpus_requested} requested, "
          f"running {n_envs} collector processes "
          f"x {frames_per_batch // n_envs} frames = {frames_per_batch} per iteration "
          f"(OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}).")
    if not IS_LINUX:
        print("[parallelism] WARNING: parallel collection is Linux-only; falling back to serial.")

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml() # 

    # Loads from "benchmarl/conf/task/Grid2OpPowerGrid/my_power_grid.yaml"
    task = G2OpPowerGridTask.MY_POWER_GRID.get_from_yaml()
    task.config["env_name"] = os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023")

    if alg == "MAPPO":
        # Loads from "benchmarl/conf/algorithm/mappo.yaml"
        algorithm_config = MappoConfig.get_from_yaml()
    elif alg == "MASAC":
        # Loads from "benchmarl/conf/algorithm/masac.yaml"
        algorithm_config = MasacConfig.get_from_yaml()
    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    
    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    with open("configs/expes_config.yaml", "r") as f:
        updates = yaml.safe_load(f)

    for config_type, config in zip(["experiment", "task", "algorithm", "model", "critic_model"],
                                    [experiment_config, task, algorithm_config, model_config, critic_model_config]):
        new_hps = updates[config_type]
        for hp in new_hps:
            if config_type == "task" and hp == "config": # In this case, we want to update the dict, not replace it
                config.config.update(new_hps[hp])
            else:
                setattr(config, hp, new_hps[hp])


    experiment_config.save_folder = os.path.join(ROOT_DIR, "saved_models")
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    experiment_config.checkpoint_at_end = save_experiment # A MASAC checkpoint is 67G

    experiment_config.max_n_frames = n_frames

    experiment_config.parallel_collection = IS_LINUX

    if alg == "MAPPO":
        experiment_config.on_policy_n_envs_per_worker = n_envs
        experiment_config.on_policy_collected_frames_per_batch = frames_per_batch
        experiment_config.on_policy_minibatch_size = experiment_config.on_policy_collected_frames_per_batch
        experiment_config.on_policy_n_minibatch_iters = MAPPO_n_episode
    elif alg == "MASAC":
        experiment_config.off_policy_n_envs_per_worker = n_envs
        experiment_config.off_policy_collected_frames_per_batch = frames_per_batch
        experiment_config.off_policy_train_batch_size = MASAC_train_batch_size
        experiment_config.off_policy_n_optimizer_steps = MASAC_n_optimizer_steps
        experiment_config.off_policy_memory_size = 500_000
    else:
        raise ValueError(f"Unknown algorithm: {alg}, possible values are 'MAPPO' or 'MASAC'")



    experiment_config.lr = lr
    experiment_config.gamma = gamma

    for i, seed in enumerate(seeds):
        print(f"Running experiment {i + 1}/{len(seeds)}.")
        train_algo(task, algorithm_config, model_config, critic_model_config, experiment_config, seed, evaluate_agents,
                   n_train_threads=n_train_threads)
        gc.collect()