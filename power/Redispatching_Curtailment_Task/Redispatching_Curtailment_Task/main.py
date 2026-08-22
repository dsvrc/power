
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
    parser.add_argument('--alg', type=str, default="MAPPO",
                        choices=["MAPPO", "MASAC", "PACT1"],
                        help="""MAPPO, MASAC, or PACT1. PACT1 is MAPPO as host with the
                                PACT-1 estimator/compensator wrapped around the env, so
                                the two share host hyperparameters exactly and 'MAPPO'
                                is the matched blind arm. (default: MAPPO)""")
    parser.add_argument('--serial', action='store_true',
                        help="""Disable parallel collection. Slow, but worker
                                exceptions surface as real tracebacks instead of
                                EOFError from a closed pipe. Use this first when a
                                run dies during collector setup.""")
    parser.add_argument('--chronics', type=str, default=None,
                        help="""Chronics season. Presets: 'summer' (.*-0[678]-.*$,
                                where thermal derating bites), 'winter'
                                (.*-(12|01|02)-.*$, the placebo arm where the
                                ampacity ratio clips to exactly 1.000), 'feb'
                                (.*-02-.*$), 'all' (no filter). Anything else is
                                used as a raw regex. Overrides
                                configs/expes_config.yaml so the summer and winter
                                arms differ only by this flag. (default: use the
                                config file)""")
    parser.add_argument('--severity', type=float, default=0.0,
                        help="""Dynamic-line-rating severity. TASK physics: applied
                                identically to MAPPO, MASAC and PACT1. 0 = static
                                ratings = stock grid2op byte for byte. 1 = realistic
                                IEEE 738 derating for the grid's own region. >1 is
                                beyond-physical and must be labelled as such.
                                (default: 0.0)""")
    parser.add_argument('--pact1_trust', type=float, default=0.90,
                        help="""PACT-1 trust prior. Initialise NEAR the optimum and let
                                the policy pull it down -- a hedged 0.5 measured ~1800
                                return worse on Ant with a correct waveform. (default: 0.90)""")
    parser.add_argument('--pact1_gate', type=str, default="prediction",
                        choices=["prediction", "prediction_only", "trace", "none"],
                        help="""Confidence gate. 'prediction' (default) = prediction gate
                                x divisor gate x readiness. 'trace' is the known-bad gate,
                                kept runnable as an ablation. (default: prediction)""")
    parser.add_argument('--pact1_max_trust', type=float, default=1.0,
                        help="""Cap on the APPLIED compensation gain (T4, III.8).
                                Compensation feeds the medium it compensates
                                against, so the individually-optimal gain sits above
                                the collective one. A Phase-1 calibration parameter
                                (II.3): sweep it, report the sweep, calibrate on one
                                seed and validate on held-out seeds. (default: 1.0,
                                i.e. uncapped)""")
    parser.add_argument('--pact1_sensor', type=str, default="max",
                        choices=["mean", "max"],
                        help="""Own-harm sensor: mean or max rho over the zone's own
                                lines. Congestion is a property of the BINDING line, so
                                'max' carries more signal; 'mean' over 23-35 lines can
                                dilute the peer effect toward zero. Both are logged
                                every row either way. (default: mean)""")
    parser.add_argument('--pact1_log', type=str, default="pact_debug.csv",
                        help="PACT-1 diagnostics CSV. Read applied_trust first.")
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
    # Explicit, not positional: unpacking vars(args).values() silently
    # misassigns every field the moment an argument is inserted.
    n_frames = args.n_frames
    lr = args.lr
    gamma = args.gamma
    frames_per_batch = args.frames_per_batch
    MAPPO_n_episode = args.MAPPO_n_episode
    MASAC_n_optimizer_steps = args.MASAC_n_optimizer_steps
    MASAC_train_batch_size = args.MASAC_train_batch_size
    seeds = args.seeds
    alg = args.alg
    save_experiment = args.save_experiment
    evaluate_agents = args.evaluate_agents
    n_envs = args.n_envs
    n_train_threads = args.n_train_threads

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

    if alg in ("MAPPO", "PACT1"):
        # PACT-1 uses MAPPO as its host, unmodified. The host algorithm is
        # never touched (III.4): PACT-1 lives entirely in an env wrapper, so
        # `--alg MAPPO` is the exactly-matched blind baseline.
        algorithm_config = MappoConfig.get_from_yaml()
    elif alg == "MASAC":
        # Loads from "benchmarl/conf/algorithm/masac.yaml"
        algorithm_config = MasacConfig.get_from_yaml()
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    # Severity is task physics: set for EVERY algorithm, never inside the
    # pact1 block. An arm-specific dial would be worthless as evidence.
    task.config["severity"] = args.severity

    if alg == "PACT1":
        task.config["pact1"] = {
            "enabled": True,
            "trust": args.pact1_trust,
            "gate": args.pact1_gate,
            "sensor": args.pact1_sensor,
            "max_trust": args.pact1_max_trust,
            "log": os.path.join(ROOT_DIR, args.pact1_log),
        }
    
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


    # Chronics season. Applied AFTER the yaml merge above, which does
    # config.config.update(...) and would otherwise silently overwrite it --
    # the run would then report one season on the command line and train on
    # another. Task physics, so it applies to every algorithm alike.
    CHRONICS_PRESETS = {
        "summer": r".*-0[678]-.*$",        # Jun-Aug: derating is active
        "winter": r".*-(12|01|02)-.*$",    # Dec-Feb: ampacity clips to 1.000
        "feb": r".*-02-.*$",               # the previous default
        "all": None,
    }
    if args.chronics is not None:
        task.config["regex_filter_chronics"] = CHRONICS_PRESETS.get(
            args.chronics, args.chronics)
    print(f"[task] severity={args.severity}  "
          f"chronics={task.config.get('regex_filter_chronics')!r}")

    experiment_config.save_folder = os.path.join(ROOT_DIR, "saved_models")
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    experiment_config.checkpoint_at_end = save_experiment # A MASAC checkpoint is 67G

    experiment_config.max_n_frames = n_frames

    experiment_config.parallel_collection = IS_LINUX and not args.serial
    if args.serial:
        print("[parallelism] SERIAL collection: worker exceptions will surface "
              "as real tracebacks instead of EOFError.")

    if alg in ("MAPPO", "PACT1"):
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