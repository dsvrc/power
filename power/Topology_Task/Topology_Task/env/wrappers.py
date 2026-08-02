import multiprocessing as mp
from collections import deque
import time

from common.imports import * 
from common.utils import split_action_tensor_dict

class AsyncMultiAgentVecEnv:
    def __init__(self, env_fns: List[Callable], context: str = "spawn"):
        self.num_envs = len(env_fns)
        ctx = mp.get_context(context)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            ctx.Process(target=self._worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.waiting = False
        self.closed = False

        # Probe observation and action spaces from first environment
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def reset(self, seed: int = None):
        for remote in self.remotes:
            remote.send(("reset", {"seed": seed}))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return self._stack_dicts(obs), list(infos)

    def step_async(self, actions: List[Dict[str, Any]]):
        # Have to split the actions dict into a list of dicts (one for each env)
        for remote, action in zip(self.remotes, split_action_tensor_dict(actions)):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, truncs, infos = zip(*results)
        self.waiting = False
        return (
            self._stack_dicts(obs),
            self._stack_dicts(rews),
            self._stack_dicts(dones),
            self._stack_dicts(truncs),
            list(infos),
        )

    def step(self, actions: List[Dict[str, Any]]):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            self.step_wait()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def _stack_dicts(self, dicts: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        # Transpose list of dicts into dict of lists, then stack
        stacked = {}
        keys = dicts[0].keys()
        for k in keys:
            stacked[k] = np.stack([d[k] for d in dicts])
        return stacked

    @staticmethod
    def _worker(remote, parent_remote, env_fn_wrapper):
        parent_remote.close()
        env = env_fn_wrapper.fn()
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == "reset":
                    obs, info = env.reset(**data)
                    remote.send((obs, info))
                elif cmd == "step":
                    observation, reward, terminated, truncated, info = env.step(data)
                    if terminated['agent_0'] or truncated['agent_0']:
                        old_observation, old_info = observation, info
                        observation, info = env.reset()
                        info["final_observation"] = old_observation
                        info["final_info"] = old_info
                    remote.send((observation, reward, terminated, truncated, info))
                elif cmd == "close":
                    remote.close()
                    break
                elif cmd == "get_spaces":
                    remote.send((env.observation_space, env.action_space))
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            print("Worker interrupted")
        finally:
            env.close()


class CloudpickleWrapper:
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle
        self.fn = pickle.loads(ob)


class RecordEpisodeStatistics(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:      # This is our case since we Record Episode Statistics only in the eval environment
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards['agent_0']
        self.episode_lengths += 1
        dones = np.logical_or(terminations['agent_0'], truncations['agent_0'])
        num_dones = np.sum(dones)

        if not num_dones and ("episode" in infos):
            del infos["episode"]
            
        if num_dones:
            infos["episode"] = {
                "r": np.where(dones, self.episode_returns, 0.0),
                "l": np.where(dones, self.episode_lengths, 0),
                "t": np.where(
                    dones,
                    np.round(time.perf_counter() - self.episode_start_times, 6),
                    0.0,
                ),
            }
            if self.is_vector_env:
                infos["_episode"] = np.where(dones, True, False)

            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
