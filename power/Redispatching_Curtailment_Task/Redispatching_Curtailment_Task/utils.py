import os
import json
import grid2op
from grid2op.utils import EpisodeStatistics


ROOT_DIR = os.getcwd()
G2OP_ENV_DIR = grid2op.get_current_local_dir()
IS_LINUX = os.name == "posix"


def _aux_get_stat_metadata(env_name, dn=True, name_stat=None):
    if os.path.isdir(env_name):
        path_env = env_name
    else:
        path_ = grid2op.get_current_local_dir()
        path_env = os.path.join(path_, env_name)
    if not os.path.exists(path_env):
        raise RuntimeError(f"The environment \"{env_name}\" does not exist.")

    path_dn = os.path.join(path_env, "_statistics_l2rpn_dn")
        
    if not os.path.exists(path_dn):
        raise RuntimeError("The folder _statistics_icaps2021_dn (or _statistics_l2rpn_dn) used for computing the score do not exist")
    path_reco = os.path.join(path_env, "_statistics_l2rpn_no_overflow_reco")
    if not os.path.exists(path_reco):
        raise RuntimeError("The folder _statistics_l2rpn_no_overflow_reco used for computing the score do not exist")
    
    if name_stat is None:
        if dn:
            path_metadata = os.path.join(path_dn, "metadata.json")
        else:
            path_metadata = os.path.join(path_reco, "metadata.json")
    else:
        path_stat = os.path.join(path_env, EpisodeStatistics.get_name_dir(name_stat))
        if not os.path.exists(path_stat):
            raise RuntimeError(f"No folder associated with statistics {name_stat}")
        path_metadata = os.path.join(path_stat, "metadata.json")
    
    if not os.path.exists(path_metadata):
        raise RuntimeError("No metadata can be found for the statistics you wanted to compute.")
    
    with open(path_metadata, "r", encoding="utf-8") as f:
        dict_ = json.load(f)
    
    return dict_


def get_env_seed(env_name: str):
    """This function ensures that you can reproduce the results of the computed scenarios.
    
    It forces the seeds of the environment, during evaluation to be the same as the one used during the evaluation of the score.
    
    As environments are stochastic in grid2op, it is very important that you use this function (or a similar one) before
    computing the scores of your agent.

    Args:
        env_name (str): The environment name on which you want to retrieve the seeds used

    Raises:
        RuntimeError: When it is not possible to retrieve the seeds (for example when the "statistics" has not been computed)

    Returns:
        List[int]: List of environment's seeds saved in the statistics folder during the score initialization
    """

    dict_ = _aux_get_stat_metadata(env_name)
    
    key = "env_seeds"
    if key not in dict_:
        raise RuntimeError(f"Impossible to find the key {key} in the metadata dictionnary. You should re run the score function.")
    
    return dict_[key]
    