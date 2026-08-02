import os
import re
import json
from collections import defaultdict
from packaging import version

from gymnasium.spaces import Discrete, Box

import grid2op
from grid2op.Chronics import MultifolderWithCache, Multifolder
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace # if we import gymnasium, GymEnv will convert to Gymnasium!   
from grid2op.multi_agent import MultiAgentEnv
from grid2op.Reward import CombinedReward
from lightsim2grid import LightSimBackend
from ray.rllib.env.multi_agent_env import MultiAgentEnv as MAEnv

from common.imports import *
from .reward import LineMarginReward, RedispRewardv1, N1ContingencyRewardv1, FlatRewardv1, DistanceRewardv1, OverloadReward

# Get the directory of the current module
ENV_DIR = os.path.dirname(__file__)

MIN_GLOP_VERSION = version.parse('1.10.4.dev1')
if version.parse(grid2op.__version__) < MIN_GLOP_VERSION:
    raise RuntimeError(f"Please upgrade to grid2op >= {MIN_GLOP_VERSION}."
                       "You might need to install it from github "
                       "`pip install git+https://github.com/rte-france/grid2op.git@dev_1.10.4`")

RHO_SAFETY_THRESHOLD = 0.95

def load_config(file_path: str) -> Dict:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    # Get the directory of the current module (__file__ contains the path of the current file)
    with open(f"{ENV_DIR}/{file_path}", 'r') as file:
        config = json.load(file)
    return config


class MAEnvWrapper(MAEnv):
    def __init__(self, args: Dict[str, Any], resume_run: bool = False, idx: int = 0, generate_class: bool = False, async_vec_env: bool = False, action_space = None, eval_env: bool = False) -> Any:
        """Create and configure a grid2op environment.

        Args:
            args: Arguments containing environment configuration parameters.
            idx: Index of the environment instance.
            resume_run: Whether to resume a previous run.
            generate_class: Whether to generate classes for asynchronous environments.
            async_vec_env: Whether the environment is asynchronous.
            action_space: A previously generated action space for the agents (to share between processes)

        Returns:
            A configured grid2op environment wrapped in a GymEnv.
        """
        super().__init__()

        config = load_config(args.env_config_path)  # Load environment configuration
        env_id = args.env_id
        env_type = args.action_type.lower()

        env_config = config['environments']
        assert env_id in env_config.keys(), f"Invalid environment ID: {env_id}. Available IDs are: {env_config.keys()}"

        env_types = ["topology", "redispatch"]
        assert env_type in env_types, f"Invalid environment type: {env_type}. Available IDs are: {env_types}"
            
        # GRID2OP
        # Create a grid2op environment with specified backend and reward structure
        # Separate rewards for the eval env for logging
        rewards = {}
        if eval_env:
            rewards['redispatchReward'] = RedispRewardv1()
            rewards['lineMarginReward'] = LineMarginReward()
            rewards['overloadReward'] = OverloadReward(constrained=args.constraints_type != 0)
            if env_type == 'topology': rewards['topologyReward'] = DistanceRewardv1()
        
        if args.n1_reward:
            rewards['n1ContingencyReward'] = N1ContingencyRewardv1(l_ids=list(range(env_config[env_id]['n_line'])), normalize=True)
        
        # With vec envs, infos return an array of dicts (one for each env) containing the rewards
        self.g2op_env = grid2op.make(
            env_config[env_id]['grid2op_id'], 
            reward_class=CombinedReward, 
            backend=LightSimBackend(),
            other_rewards=rewards,
            chronics_class=Multifolder if args.optimize_mem else MultifolderWithCache 
        ) 

        ###
        #print(self.g2op_env.chronics_handler.max_episode_duration())

        self.g2op_env.seed(args.seed+idx)
        self.g2op_env.chronics_handler.seed(args.seed+idx)
        self.g2op_env.chronics_handler.shuffle()

        if args.optimize_mem: self.g2op_env.chronics_handler.set_chunk_size(100)    # Instead of loading all episode data, get chunks of 100

        # Assign a filter (e.g., use only chronics that have "december" in their name) to reduce memory footprint
        #self.g2op_env.chronics_handler.set_filter(lambda x: re.match(".*0000.*", x) is not None)
        # Create the cache; otherwise it'll only load the first scenario
        # TODO seed setup does not work with this - if we print the seeds, they are different (but agents' obs across envs are still the same)
        self.g2op_env.chronics_handler.reset()
            
        #print(self.g2op_env.chronics_handler.max_episode_duration())

        cr = self.g2op_env.get_reward_instance()  # Initialize the combined reward instance
        # Per step (cumulative) positive reward for staying alive; reaches 1 at the end of the episode
        #cr.addReward("IncreasingFlatReward", 
                    #IncreasingFlatRewardv1(per_timestep=1/g2op_env.chronics_handler.max_episode_duration()),
        #            1.0)
        cr.addReward("FlatReward", FlatRewardv1(per_timestep=1), 1.0)
        if not eval_env:
            #if env_type == 'topology': cr.addReward("TopologyReward", DistanceRewardv1(), 0.3)   # = 0 if topology is the original one, -1 if everything changed
            cr.addReward("redispatchReward", RedispRewardv1(), 1.0)  # Custom one, see common.rewards
            #cr.addReward("LineMarginReward", LineMarginReward(), 0.5)  # Custom one, see common.rewards
        if args.constraints_type != 2:
            cr.addReward("overloadReward", OverloadReward(constrained=args.constraints_type != 0), 1.0)  # Custom one, see common.rewards

        cr.initialize(self.g2op_env)  # Finalize the reward setup

        if generate_class:
            self.g2op_env.generate_classes()
            print("Class generated offline for AsyncVecEnv execution")
            quit()
        
        # MARL
        agent_ids = range(len(config['environments'][env_id]['agent_stations']) if args.difficulty == 0 else self.g2op_env.n_sub) 
        if args.difficulty == 0:
            self.action_domains = {f"agent_{idx}": 
                                config['environments'][env_id]['agent_stations'][idx]
                                for idx in agent_ids
            }
        elif args.difficulty == 1:
            self.action_domains = {f"agent_{idx}": [idx] for idx in agent_ids}
        else:
            raise NotImplementedError("There are only 2 difficulty levels!")

        if args.decentralized:
            self.observation_domains = self.action_domains
        else:
            self.observation_domains = {f"agent_{idx}": list(range(self.g2op_env.n_sub))
                                        for idx in agent_ids
            }

        self.g2op_ma_env = MultiAgentEnv(self.g2op_env,
                        action_domains=self.action_domains,
                        observation_domains=self.observation_domains)

        self._agent_ids = set(self.g2op_ma_env.agents)
        self._agent_ids = self.g2op_ma_env.agents
        self.g2op_ma_env.seed(args.seed+idx)

        # Prepare action and observation spaces
        state_attrs = config['state_attrs']
        obs_attrs = state_attrs['default']
        if env_config[env_id]['maintenance']: obs_attrs += state_attrs['maintenance']

        if env_type == 'topology': 
            obs_attrs += state_attrs['topology']
            obs_attrs += state_attrs['redispatch']
            if env_config[env_id]['renewable'] : 
                obs_attrs += state_attrs['curtailment']
            if env_config[env_id]['battery']:
                obs_attrs += state_attrs['storage']
        else:
            raise NotImplementedError("Redispatching environments are not implemented yet!")
        
        # MARL spaces
        self._aux_observation_space = {
            agent_id : BoxGymObsSpace(self.g2op_ma_env.observation_spaces[agent_id],
                                      attr_to_keep=obs_attrs,
                                      )
            for agent_id in self.g2op_ma_env.agents
        }

        # to avoid "weird" pickle issues
        self.observation_space = {
            agent_id : Box(low=self._aux_observation_space[agent_id].low,
                           high=self._aux_observation_space[agent_id].high,
                           dtype=self._aux_observation_space[agent_id].dtype)
            for agent_id in self.g2op_ma_env.agents
        }

        # raise alert or alarm is not supported by ALL_ATTR_FOR_DISCRETE nor ATTR_DISCRETE
        act_attr_to_keep = ['change_bus', 'change_line_status']
        if env_type == 'topology': 
            self._conv_action_space = {
                    agent_id : DiscreteActSpace(self.g2op_ma_env.action_spaces[agent_id], attr_to_keep=act_attr_to_keep)
                    for agent_id in self.g2op_ma_env.agents
                }
                
            #to avoid "weird" pickle issues
            self.action_space = {
                agent_id : Discrete(n=self._conv_action_space[agent_id].n)
                for agent_id in self.g2op_ma_env.agents
            }
        else:
            raise NotImplementedError("Make the implementation in this case")

        self.constraints_type = args.constraints_type

        self.norm_obs = args.norm_obs
        if self.norm_obs:
            self.epsilon = 1e-8
            self.obs_stats = defaultdict(lambda: {
                "count": 1e-4,
                "mean": None,
                "var": None,
            })

        self.use_heuristic = args.use_heuristic
    
    @property
    def _risk_overflow(self) -> bool:
        """Check if the maximum rho value exceeds the safety threshold."""
        return self.g2op_ma_env._cent_env.current_obs.rho.max() >= RHO_SAFETY_THRESHOLD

    @property  
    def _obs(self) -> np.ndarray:
        """Get the current observation from the Grid2Op environment."""
        return self.g2op_ma_env._cent_env.current_obs
    
    def _get_idle_actions(self) -> List:
        """Return an empty list of actions if risk overflow, otherwise return a default action."""   
        if self._risk_overflow: return []
        return {
                agent_id : self._conv_action_space[agent_id].from_gym(0)
                for agent_id in self._agent_ids
            }
    
    def apply_idle_actions(self) -> Tuple[float, bool, Dict]:
        """Apply heuristic actions until a risky situation or episode end.

        Returns:
            A tuple containing the cumulative heuristic reward, a boolean indicating
            if the episode is done, and additional info.
        """
        use_heuristic, heuristic_reward = True, 0
        done, info = False, {}
       
        while use_heuristic:
            g2o_actions = self._get_idle_actions()
            if not g2o_actions: break

            obs, reward, done, info = self.g2op_ma_env.step(g2o_actions)
            
            heuristic_reward += reward['agent_0']

            if done['agent_0'] or self._risk_overflow:   # Resume the agent if in a risky situation
                use_heuristic = False
                break

        return obs, heuristic_reward, done, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        done = True     # It could happen that a heuristic episode ends in the reset step
        while done:
            obs = self.g2op_ma_env.reset()  # reset the underlying multi agent environment

            if self.use_heuristic and not self._risk_overflow: 
                obs, _, done, _ = self.apply_idle_actions()

            done = done['agent_0'] if isinstance(done, Dict) else False  # Manage the case when self._risk_overflow directly after reset

        return self._format_obs(obs), {}
        
    def seed(self, seed):
        return self.g2op_ma_env.seed(seed)
    
    def _format_obs(self, grid2op_obs):
        gym_obs = {
            agent_id : self._aux_observation_space[agent_id].to_gym(grid2op_obs[agent_id])
            for agent_id in self.g2op_ma_env.agents
        }
        if self.norm_obs:
            self._update_stats(gym_obs)
            gym_obs = self._normalize(gym_obs)

        # return the proper dictionnary
        return gym_obs

    def _update_stats(self, obs):
        for agent_id, ob in obs.items():
            stats = self.obs_stats[agent_id]
            if stats["mean"] is None:
                stats["mean"] = np.zeros_like(ob)
                stats["var"] = np.ones_like(ob)
            stats["count"] += 1
            delta = ob - stats["mean"]
            stats["mean"] += delta / stats["count"]
            delta2 = ob - stats["mean"]
            stats["var"] += delta * delta2

    def _normalize(self, obs):
        norm_obs = {}
        for agent_id, ob in obs.items():
            stats = self.obs_stats[agent_id]
            var = stats["var"] / stats["count"]
            var = np.maximum(var, self.epsilon)  # avoid sqrt of negative or zero
            std = np.sqrt(var)
            normed = (ob - stats["mean"]) / std
            normed = np.where(np.isfinite(normed), normed, 0.0)  # replace NaN/inf with 0
            norm_obs[agent_id] = normed
        return norm_obs
    
    def _get_grid2op_act(self, actions):
        return {
            agent_id : self._conv_action_space[agent_id].from_gym(actions[agent_id] if actions else 0)
            for agent_id in self.g2op_ma_env.agents
        }

    def step(self, actions):       
        # convert the action to grid2op
        grid2op_act = self._get_grid2op_act(actions)
            
        # do a step in the underlying multi agent environment
        obs, r, done, info = self.g2op_ma_env.step(grid2op_act)

        if self.use_heuristic and not done['agent_0'] and not self._risk_overflow:
            obs, heuristic_reward, done, info = self.apply_idle_actions()
            r = {k: v + heuristic_reward for k, v in r.items()}

        self._get_cost(done, info)

        # Retrieve the observation in the proper form
        gym_obs = self._format_obs(obs)
        
        truncateds = {k: False for k in self.g2op_ma_env.agents}    # TODO truncation is not used in g2o

        return gym_obs, r, done, truncateds, info

    def _get_cost(self, done, info): 
        if self.constraints_type == 1:  # TODO add check on n° steps (if it's done but the grid survived for the entire episode, it's not a constraint violation)
            if done['agent_0']: info['cost'] = 1
            else: info['cost'] = 0
        elif self.constraints_type == 2:
            n_disconnections = sum(~self.g2op_ma_env._cent_env.current_obs.line_status)
            n_overloads = 0     # If it's game over and the grid is disconnected (i..e, n_disconnections = n_lines), then we cannot get the thermal limit and we don't have to compute n_overloads

            if not done['agent_0']:
                ampere_flows = np.abs(self.g2op_ma_env._cent_env.backend.get_line_flow())
                thermal_limits = np.abs(self.g2op_ma_env._cent_env.get_thermal_limit())        
                margin = thermal_limits - ampere_flows
                n_overloads = len(margin[margin < 0])
            
            info['cost'] = n_disconnections + n_overloads
        else:
            return
