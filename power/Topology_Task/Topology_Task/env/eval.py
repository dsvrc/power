from collections import deque

from common.imports import *
from common.logger import Logger
from common.utils import cast_np_to_tensors, stack_agent_obs_by_env
from .utils import MAEnvWrapper
from .wrappers import RecordEpisodeStatistics

class Evaluator:
    """Evaluator class for evaluating a reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, args: Dict[str, Any], logger: Logger, device: th.device) -> None:
        """Initialize the Evaluator with the given arguments, logger, and device.

        Args:
            args: Arguments containing environment configuration.
            logger: Logger for storing evaluation metrics.
            device: Device to run the model on.
        """
 
        self.env = RecordEpisodeStatistics(MAEnvWrapper(args, eval_env=True))  # Initialize synchronized vector environment

        self.max_steps = self.env.g2op_env.chronics_handler.max_episode_duration()  # Get max episode duration
        self.logger = logger  # Logger for evaluation metrics
        self.device = device  # Device for model inference
        self.env_id = args.env_id
        # Fix n° rewards based on the env specs (for simplifying logging ops)
        # The reward returned by the env is an increasing survival reward
        self.reward_tags = ['Redispatch Reward', 'Line Margin Reward', 'Overload Reward']
        if args.action_type == 'topology': self.reward_tags += ['Topology Reward']      
        if args.n1_reward: self.reward_tags += ['N1 Reward']
        if self.env_id == 'bus118': self.reward_tags.append([])     # l2rpn_idf_2023 (bus118) returns an additional 0 reward value; we fix it with an empty metrics
        self.use_heuristic = args.use_heuristic
        #if self.use_heuristic: self.env.set_n_rewards(len(self.reward_tags))
            
    def evaluate(self, glob_step: int, actors: Dict, eval_ep: int = 3) -> None:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """
        
        ep_survivals: Deque[float] = deque(maxlen=eval_ep)  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))

        obs, info = self.env.reset()
        obs = cast_np_to_tensors(obs, self.device)        
        #if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        action = {}
        while len(ep_survivals) < eval_ep:

            for agent, model in actors.items(): action[agent] = model.get_eval_action(obs[agent])

            next_obs, _, _, _, info = self.env.step(action)

            obs = cast_np_to_tensors(next_obs, self.device)        
            if not self.use_heuristic: ep_rewards += list(info['agent_0']['rewards'].values())
            # Record rewards for plotting purposes
            if "episode" in info :   # Denote end of an episode
                ep_survivals.append(self.env.g2op_ma_env._cent_env.nb_time_step/self.max_steps)
                if not self.use_heuristic: ep_returns.append(ep_rewards)
                obs, _ = self.env.reset()
                obs = cast_np_to_tensors(obs, self.device)        
                if not self.use_heuristic: ep_rewards = np.zeros(len(self.reward_tags))

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals)/eval_ep
        avg_return = [sum(r)/eval_ep for r in zip(*ep_returns)]

        # Log the metrics if logger is available
        if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, self.reward_tags if self.env_id != 'bus118' else self.reward_tags[:-1])

        print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}")

class CMDPEvaluator(Evaluator):
    """Evaluator class for evaluating a constrained reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, args: Dict[str, Any], logger: Logger, device: th.device) -> None:
        super().__init__(args, logger, device)
            
    def evaluate(self, glob_step: int, actors: Dict, eval_ep: int = 3) -> None:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """
        
        ep_survivals: Deque[float] = deque(maxlen=eval_ep)  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store returns of episodes
        ep_cost_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store cost returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))
        ep_costs = 0

        obs, info = self.env.reset()
        obs = cast_np_to_tensors(obs, self.device)        
        if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        action = {}
        while len(ep_survivals) < eval_ep:

            for agent, model in actors.items(): action[agent] = model.get_eval_action(obs[agent])

            next_obs, _, _, _, info = self.env.step(action)

            obs = cast_np_to_tensors(next_obs, self.device)        
            ep_rewards += list(info['agent_0']['rewards'].values())
            ep_costs += info['cost']

            # Record rewards for plotting purposes
            if "episode" in info :   # Denote end of an episode
                ep_survivals.append(self.env.g2op_ma_env._cent_env.nb_time_step/self.max_steps)
                ep_returns.append(ep_rewards)
                ep_cost_returns.append(ep_costs)

                obs, _ = self.env.reset()
                obs = cast_np_to_tensors(obs, self.device)        
                ep_rewards = np.zeros(len(self.reward_tags))
                ep_costs = 0

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals)/eval_ep
        avg_return = [sum(r)/eval_ep for r in zip(*ep_returns)]
        avg_cost_return = [sum(ep_cost_returns)/eval_ep]

        # Log the metrics if logger is available
        if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, avg_cost_return, self.reward_tags if self.env_id != 'bus118' else self.reward_tags[:-1])

        print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}")
