
import numpy as np
from abc import abstractmethod
from typing import List, Optional, Tuple, Dict

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

from benchmarl.environments.G2OpPowerGrid.PZMAEnvWithHeuristics import PZMAEnvWithHeuristics as EnvWithHeuristics

class APICustomAgent(BaseAgent):
    """
    This can then be used as a "regular" grid2op agent, in a runner, grid2viz, grid2game etc.
    It adapts the l2rpn_baselines.utils.GymEnv class.

    Use it only with a trained agent. It does not provide the "save" method and
    is not suitable for training.
    
    .. note::
        To load a previously saved agent the function `APICustomAgent.load` will be called
        and you must provide the `nn_path` keyword argument.
        
        To build a new agent, the function `APICustomAgent.build` is called and
        you must provide the `nn_kwargs` keyword argument.
    
    
    Notes
    -----
    The main goal of this class is to be able to use "heuristics" (both for training and at inference time) quite simply
    and with out of the box support of external libraries.
    
    All top performers in all l2rpn competitions (as of writing) used some kind of heuristics in their agent (such as: 
    "if a powerline can be reconnected, do it" or "do not act if the grid is not in danger"). This is why we made some 
    effort to develop a generic class that allows to train agents directly using these "heuristics".
    
    This features is split in two parts:
    
    - At training time, the "*heuristics*" are part of the environment. The agent will see only observations that are relevant
      to it (and not the stat handled by the heuristic.)
    - At inference time, the "*heuristics*" of the environment used to train the agent are included in the "agent.act" function.
      If a heuristic has been used at training time, the agent will first "ask" the environment is a heuristic should be
      performed on the grid (in this case it will do it) otherwise it will ask the underlying neural network what to do.
    
    """
    def __init__(self,
                 g2op_action_space,
                 *,  # to prevent positional argument
                 nn_path=None,
                 nn_kwargs=None,
                 apienv=None,
                 _check_both_set=True,
                 _check_none_set=True):
        super().__init__(g2op_action_space)
        
        self._has_heuristic : bool = False
        self.apienv : Optional[EnvWithHeuristics] = apienv
        self._action_list : Optional[List] = None
        
        if self.apienv is not None and isinstance(self.apienv, EnvWithHeuristics):
            self._has_heuristic = True
            self._action_list = []
            
        if _check_none_set and (nn_path is None and nn_kwargs is None):
            raise RuntimeError("Impossible to build an APICustomAgent without providing at "
                               "least one of `nn_path` (to load the agent from disk) "
                               "or `nn_kwargs` (to create the underlying agent).")
        if _check_both_set and (nn_path is not None and nn_kwargs is not None):
            raise RuntimeError("Impossible to build an APICustomAgent by providing both "
                               "`nn_path` (*ie* you want load the agent from disk) "
                               "and `nn_kwargs` (*ie* you want to create the underlying agent from these "
                               "parameters).")
        if nn_path is not None:
            self._nn_path = nn_path
        else:
            self._nn_path = None
            
        if nn_kwargs is not None:
            self._nn_kwargs = nn_kwargs
        else:
            self._nn_kwargs = None
        
        self.nn_model = None
        if nn_path is not None:
            self.load()
        else:
            self.build()
            
    @abstractmethod
    def get_act(self, api_obs, reward, done):
        """
        retrieve the action from the NN model
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load the NN model
        
        ..info:: Only called if the agent has been build with `nn_path` not None and `nn_kwargs=None`
        """
        pass
    
    @abstractmethod
    def build(self):
        """
        Build the NN model.
        
        ..info:: Only called if the agent has been build with `nn_path=None` and `nn_kwargs` not None
        """
        pass

    @abstractmethod
    def _to_api_obs(self, observation: BaseObservation) -> Tuple[np.ndarray, Dict]:
        pass

    @abstractmethod
    def _from_api_act(self, api_action: np.ndarray) -> BaseAction:
        pass

    def clean_heuristic_actions(self, observation: BaseObservation, reward: float, done: bool) -> None:
        """This function allows to cure the heuristic actions. 
        
        It is called at each step, just after the heuristic actions are computed (but before they are selected).
        
        It can be used, for example, to reorder the `self._action_list` for example.

        It is not used during training.
        
        Args:
            observation (BaseObservation): The current observation
            reward (float): the current reward
            done (bool): the current flag "done"
        """
        pass
    
    def act(self, observation: BaseObservation, reward: float, done: bool) -> BaseAction:
        """This function is called to "map" the grid2op world
        into a usable format by a neural networks (for example in a format
        usable by ray/rllib or benchmarl)

        Parameters
        ----------
        observation : BaseObservation
            The grid2op observation
        reward : ``float``
            The reward
        done : function
            the flag "done"

        Returns
        -------
        BaseAction
            The action taken by the agent, in a form of a grid2op BaseAction.
        
        Notes
        -------
        In case your "real agent" wants to implement some "non learned" heuristic,
        you can also put them here.
        
        In this case the "benchmarl agent" will only be used in particular settings.
        """
        grid2op_act = None
        
        # heuristic part
        if self._has_heuristic:
            if not self._action_list:
                # the list of actions is empty, i querry the heuristic to see if there's something I can do
                self._action_list = self.apienv.heuristic_actions(observation, reward, done, {})
                
            self.clean_heuristic_actions(observation, reward, done)
            if self._action_list:
                # some heuristic actions have been selected, i select the first one
                grid2op_act = self._action_list.pop(0)
        
        # the heursitic did not select any actions, then ask the NN to do one !
        if grid2op_act is None:
            api_obs = self._to_api_obs(observation)
            api_act = self.get_act(api_obs, reward, done)
            grid2op_act = self._from_api_act(api_act)
            
            # fix the action if needed (for example by limiting curtailment and storage)
            if self._has_heuristic:
                grid2op_act = self.apienv.fix_action(grid2op_act, observation)
            
        return grid2op_act
    