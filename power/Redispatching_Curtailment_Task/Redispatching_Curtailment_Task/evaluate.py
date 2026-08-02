
import os
import json
import argparse
import shutil
import numpy as np
import grid2op
from grid2op.Action import PlayableAction
from lightsim2grid import LightSimBackend
from utils import *
from grid2op.utils import ScoreL2RPN2020

def cli():
    parser = argparse.ArgumentParser(description="Evaluate your agent.")
    parser.add_argument('--agent_name', type=str, 
                        help="Name of the agent to evaluate. It should correspond to a folder name in saved_models.")
    parser.add_argument('--env_name', type=str, default=os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023_test_new"),
                        help="Name of the environment on which you evaluate. (default: l2rpn_idf_2023_test_new)")
    parser.add_argument('--n_episode', type=int, default=-1,
                        help="Number of episodes to evaluate. If -1, it evaluates the 2 first years. (default: -1)")
    parser.add_argument('--save_results', type = int, choices=[0, 1], default=1,
                        help="Whether or not to save results. 1 for yes 0 for no. (default: 1)")
    parser.add_argument('--gpu_device', type=int, choices=[-1, 0, 1, 2, 3], default=-1,
                        help="Index of the gpu to lauch on. Not used on windows or if set to -1. (default: -1)")
    parser.add_argument('--verbose', type = int, choices=[0, 1, 2], default=2,
                        help="Score's verbosity, 0 for nothing, 1 for explanainations, 2 to add loading bars. (default: 2)")
    return parser.parse_args()

def evaluate(agent, 
             agent_name, 
             env_name=os.path.join(G2OP_ENV_DIR,"l2rpn_idf_2023_test_new"), 
             n_episode=-1, 
             results_path_agents=os.path.join(ROOT_DIR, "agents_results"), 
             verbose=2):
    env = grid2op.make(env_name,
                            action_class=PlayableAction,
                            backend=LightSimBackend(),
                            )

    n_chronics = env.chronics_handler.available_chronics().shape[0]
    if n_episode == -1:
        # Case 1: We evaluate only the 2 first years
        if n_chronics % 52 != 0:
            raise ValueError("""The number of chronics should be a multiple of 52 because there is 52 weeks in
                              a year. It means at least a week is missing in your data.""")
        n_years = n_chronics // 52
        years_to_evaluate = (0, 1) # Be careful, the order is 0, 1, 10, 11, 2, 3, ..., 9. So 2 refers to year 10.
        episode_id = [k * n_years + i for k in range(52) for i in years_to_evaluate] # TODO Doesn't work if a chronic is missing
    else:
        # Case 2 : We evaluate the n_episode first scenarios
        episode_id = range(min(n_episode, n_chronics))
    n_episode = len(episode_id)

    # env_seeds = [334206740, 366244288, 18817077, 1248524774, 1594006673, 2080723336, 358888931, 2092397877, 447027417, 
    #              1855732691, 1434042798, 221113676, 1693162652, 1216701780, 550735510, 1186852924, 208368583, 1440349688, 
    #              1604369804, 962316843, 492280266, 81928991, 837759323, 1106008516, 453569866, 795616554, 1914220920, 
    #              828971598, 1063489233, 1326380595, 1919368776, 917797430, 1431712628, 135050026, 1892797774, 1901646481, 
    #              1078202844, 683893452, 515194455, 936418504, 1710307152, 1093711678, 1621632363, 1288439232, 703680654, 
    #              2112655567, 1739546528, 523909909, 728200700, 957373352, 1247825955, 1383142112, 595224264, 1292708494, 
    #              1841255586, 35858627, 1233617504, 1541122331, 1086259130, 1519408243, 121834766, 306026125, 292307791, 1874557471, 
    #              642744267, 1536089306, 962257878, 1471791410, 1606082079, 1985286260, 427797647, 1632880723, 875998721, 583804985, 
    #              1040467121, 1950619580, 2114516141, 1374461419, 1298492240, 1807051232, 1994706339, 2142396189, 841792948, 
    #              923240700, 264317719, 1617243183, 1287747104, 1579668088, 1366987405, 412374990, 94200414, 314748399, 1797532073,
    #              1478766375, 1010185260, 1012972699, 42420097, 432665932, 728885748, 1530950505, 1276552691, 1946023462, 2784469, 1302957273]

    env_seeds = get_env_seed(env_name)[:n_episode]
    agent_seeds=[0 for _ in range(n_episode)]
    
    if results_path_agents is not None:
        results_path_agent = os.path.join(results_path_agents, agent_name)
        os.makedirs(results_path_agent, exist_ok=False)
        shutil.rmtree(os.path.abspath(results_path_agent), ignore_errors=True)

    # runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=global_agent)
    # res = runner.run(nb_episode=n_episode, pbar=False, 
    #                 # add_detailed_output=True,
    #                 env_seeds=env_seeds[:n_episode],
    #                 agent_seeds=agent_seeds[:n_episode],
    #                 episode_id=episode_id, 
    #                 path_save=results_path_agent if save_results else None,
    #                 )

    my_score = ScoreL2RPN2020(env,
                                nb_scenario=n_episode,
                                env_seeds=env_seeds,
                                agent_seeds=agent_seeds,
                                verbose=verbose,
                                )


    _, ts_survived, _ = my_score.get(agent, path_save=results_path_agent)
    print(ts_survived)

    print("mean +- std:", np.mean(ts_survived), "+-", np.std(ts_survived))
    print("median:", np.median(ts_survived))

    result_dict = {"nb_scenario": n_episode,
                        "env_seeds": env_seeds,
                        "agent_seeds": agent_seeds}
    result_dict["mean_timestep_played"] = np.mean(ts_survived)
    result_dict["median_timestep_played"] = np.median(ts_survived)
    result_dict["ts_survived"] = ts_survived

    if results_path_agent is not None:
        with open(os.path.join(results_path_agent, 'results.json'), 'w') as file:
            json.dump(result_dict, file, indent = 4)


if __name__ == "__main__":

    from BMMAAgent import BMMAAgent
    from grid2op.Agent import DoNothingAgent, RecoPowerlineAgent
    import torch
    import multiprocessing as mp

    args = cli()
    ### Unpacking arguments
    args_dict = vars(args)
    agent_name, env_name, n_episode, save_results, gpu_device, verbose = args_dict.values()

    if IS_LINUX:
        mp.set_start_method("fork", force=True)
        if gpu_device != -1:
            torch.cuda.set_device(gpu_device)

    env = grid2op.make(env_name)

    save_path_agents = os.path.join(ROOT_DIR, "saved_models")
    results_path_agents = os.path.join(ROOT_DIR, "agents_results")

    if agent_name=="DoNothing":
        agent = DoNothingAgent(env.action_space)
    elif agent_name=="RecoPowerline":
        agent = RecoPowerlineAgent(env.action_space)
    else:
        agent = BMMAAgent(env.action_space, 
                        nn_path=os.path.join(save_path_agents, agent_name))
        if IS_LINUX:
            for p in mp.active_children():
                p.terminate()
        
    results_path_agents = results_path_agents if save_results else None
    evaluate(agent, 
             agent_name=agent_name,
             env_name=env_name, 
             n_episode=n_episode, 
             results_path_agents=results_path_agents, 
             verbose=2)

                        

    



