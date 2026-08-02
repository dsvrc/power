
import os
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from benchmarl.hydra_config import reload_experiment_from_file
from APICustomAgent import APICustomAgent, EnvWithHeuristics
import warnings

class BMMAAgent(APICustomAgent):
    """This class represents the agent (directly usable with grid2op framework).

    This agents uses the benchmarl neural network to take decisions on the grid.
    
    To be built, it requires:
    
    - `g2op_action_space`: a grid2op action space (used for initializing the grid2op agent)
    
    It can also accept different types of parameters:
    
    - `nn_kwargs`: the parameters used to build the benchmark "trainer"
    - `nn_path`: the path where the neural network can be loaded from
    
    For this class, providing strictly one of the two later is mandatory.
                                   
    """
    def __init__(self,
                 g2op_action_space,
                 nn_kwargs=None,
                 nn_path=None,
                 iter_num=None,
                 ):
        self._iter_num = iter_num
        super().__init__(g2op_action_space, nn_path=nn_path, nn_kwargs=nn_kwargs)
        

        if isinstance(self.apienv, EnvWithHeuristics):
            self._has_heuristic = True
            self._action_list = []

        print("***AGENT INITIALIZED***")

    def _to_api_obs(self, g2op_obs):
        return self.apienv._to_gym_obs(g2op_obs)

    def _from_api_act(self, api_action):
        return self.apienv._from_gym_act(api_action)
    
    def _deterministic_forward(self, policy: TensorDictSequential, td_in: TensorDict) -> TensorDict:
            td = td_in.clone()
            with torch.no_grad():
                for actor in policy.module:  # ModuleList of ProbabilisticActors
                    td = actor(td)
            return td
        
    def get_act(self, api_obs, reward, done):
        """Retrieve the api action from the api observation and the reward. 

        Parameters
        ----------
        api_obs : api observation
            The api observation
        reward : ``float``
            the current reward
        done : ``bool``
            whether the episode is over or not.

        Returns
        -------
        api action
            The api action, that is processed in the :func:`BMMAAgent.act`
            to be used with grid2op
        """
        # --- Convert obs dict to TensorDict input ---
        obs_tensor_dict = TensorDict(
            {
                (agent, "observation"): torch.tensor(api_obs_a, dtype=torch.float32).unsqueeze(0)  # Add batch dim
                for agent, api_obs_a in api_obs.items()
            },
            batch_size=[1],
        )

        # --- Deterministic inference ---
        td_out = self._deterministic_forward(self.nn_model, obs_tensor_dict)

        # --- Extract actions into a clean output dict ---
        api_act = {
            agent: td_out[(agent, "action")].squeeze(0).cpu().numpy()
            for agent in self.apienv.agents
        }

        return api_act

    def load(self, experiment=None):
        """
        Load the NN model.
        
        """
        if experiment is None:
            checkpoints_dir = os.path.join(self._nn_path, "checkpoints")
            if self._iter_num is None:
                checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith("checkpoint_")]
                iter_max = max([int(f.split("_")[-1].split(".")[0]) for f in checkpoints])
                self._iter_num = iter_max
            else:
                self._iter_num = int(self._iter_num)
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{self._iter_num}.pt")
            experiment = reload_experiment_from_file(checkpoint_path)
        self.apienv = experiment.test_env.base_env._env
        # if isinstance(self.apienv, EnvWithHeuristics):
        self._has_heuristic = True
        self._action_list = []
        self.nn_model = experiment.policy
        
    def build(self):
        """Create the underlying NN model from scratch.            
        """
        # raise NotImplementedError("Not implemented yet.")
        warnings.warn("Building the BMMAAgent from scratch is not implemented yet.")

if __name__ == "__main__":
    import grid2op
    from lightsim2grid import LightSimBackend
    from utils import ROOT_DIR, G2OP_ENV_DIR
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    
    # env_name = os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023")  # or any other name
    env_name = os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023_test_new")  # or any other name
    
    # create the grid2op environment
    env = grid2op.make(env_name, backend=LightSimBackend())
    
    saved_agents_path = os.path.join(ROOT_DIR, "saved_agents")

    expe_path = os.path.join(saved_agents_path, "mappo_my_power_grid_mlp__da4d7f18_25_04_19-18_33_48")

    # create a grid2gop agent based on that (this will reload the save weights)
    grid2op_agent = BMMAAgent(env.action_space,
                                nn_path=expe_path,  # don't load it from anywhere
                                # iter_num=1280,  # load the last iteration
                                )
    for p in mp.active_children():
        p.terminate()
    print("***AGENT CREATED***")
    # use it
    obs = env.reset()
    reward = env.reward_range[0]
    done = False
    grid2op_act = grid2op_agent.act(obs, reward, done)
    obs, reward, done, info = env.step(grid2op_act)

    print("***THE END***")
    