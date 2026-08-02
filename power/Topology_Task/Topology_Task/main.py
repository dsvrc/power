from time import time

from alg.qplex.core import QPLEX
from alg.lagr_mappo.core import LagrMAPPO
from alg.mappo.core import MAPPO
from common.checkpoint import MAPPOCheckpoint, QPLEXCheckpoint, LagrMAPPOCheckpoint
from common.imports import *
from common.utils import set_random_seed, set_torch, str2bool
from env.config import get_env_args
from env.utils import MAEnvWrapper
from env.wrappers import AsyncMultiAgentVecEnv

# Dictionary mapping algorithm names to their corresponding classes
ALGORITHMS: Dict[str, Type[Any]] = {'QPLEX': QPLEX, 'MAPPO': MAPPO, 'LAGRMAPPO': LagrMAPPO}

def main(args: Namespace) -> None:
    """
    Main function to run the RL algorithms based on the provided arguments.

    Args:
        args (Namespace): Command line arguments parsed by argparse.

    Raises:
        AssertionError: If time limit exceeds 2800 minutes or if number of environments is less than 1.
        AssertionError: If the specified algorithm is not supported.
    """
    assert args.time_limit <= 2800, f"Invalid time limit: {args.time_limit}. Timeout limit is : 2800"
    start_time = time()
    
    # Update args with environment arguments
    args = ap.Namespace(**vars(args), **vars(get_env_args()))
    assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"
    
    alg = args.alg.upper()
    assert alg in ALGORITHMS.keys(), f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
    if (alg == "LAGRMAPPO" and args.constraints_type == 0) or (alg != "LAGRMAPPO" and args.constraints_type in [1, 2]):
        raise ValueError("Check the constrained version of the alg/env!")

    run_name = args.resume_run_name if args.resume_run_name \
        else f"{args.alg}_{args.env_id}_{"T" if args.action_type == "topology" else "R"}_{args.seed}_{args.difficulty}_{"H" if args.use_heuristic else ""}_{"I" if args.heuristic_type == "idle" else ""}_{"C1" if args.constraints_type == 1 else "C2" if args.constraints_type == 2 else ""}_{int(time())}_{np.random.randint(0, 50000)}"

    # Initialize the appropriate checkpoint based on the algorithm
    if alg == 'MAPPO': checkpoint = MAPPOCheckpoint(run_name, args)
    elif alg == 'QPLEX': checkpoint = QPLEXCheckpoint(run_name, args)
    elif alg == 'LAGRMAPPO': checkpoint = LagrMAPPOCheckpoint(run_name, args)

    else:
        pass  # This case should not occur due to earlier assertion

    # Set random seed and Torch configuration
    set_random_seed(args.seed)
    set_torch(args.n_threads, args.th_deterministic, args.cuda)
    
    # Resume run if checkpoint was resumed
    if checkpoint.resumed: args = checkpoint.loaded_run['args']
  
    env_fns = [lambda i=i: MAEnvWrapper(args, idx=i) for i in range(args.n_envs)]
    envs = AsyncMultiAgentVecEnv(env_fns)
    
    # Run the specified algorithm
    ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint)
        
if __name__ == "__main__":
    # mp.get_context("forkserver")
    parser = ap.ArgumentParser()

    # Cluster
    parser.add_argument("--time-limit", type=float, default=1300, help="Time limit for the action ranking")
    parser.add_argument("--checkpoint", type=str2bool, default=False, help="Toggles checkpoint.")
    parser.add_argument("--resume-run-name", type=str, default='', help="Run name to resume")

    # Reproducibility [MAPPO, QPLEX, LAGRMAPPO]
    parser.add_argument("--alg", type=str, default='MAPPO', help="Algorithm to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Logger
    parser.add_argument("--verbose", type=str2bool, default=True, help="Toggles prints")
    parser.add_argument("--exp-tag", type=str, default='', help="Tag for logging the experiment")
    parser.add_argument("--track", type=str2bool, default=False, help="Tag for logging the experiment")
    parser.add_argument("--wandb-project", type=str, default="marl2grid_test", help="Wandb's project name.")
    parser.add_argument("--wandb-entity", type=str, default="emarche", help="Entity (team) of wandb's project.")
    parser.add_argument("--wandb-mode", type=str, default="online", help="Online or offline wandb mode.")

    # Torch
    parser.add_argument("--th-deterministic", type=str2bool, default=True, help="Enable deterministic in Torch.")
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=4, help="Max number of torch threads.")

    main(parser.parse_known_args()[0])
