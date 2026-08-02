from time import time

from stable_baselines3.common.buffers import ReplayBuffer

from .agent import QNetwork, QPLEXMixer
from .config import get_alg_args
from .memory import Buffer
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from common.utils import cast_np_to_tensors, stack_agent_obs_by_env
from env.eval import Evaluator

def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Calculate the linear schedule for epsilon decay.

    Args:
        start_e: Starting epsilon value.
        end_e : Ending epsilon value.
        duration: Total duration over which epsilon decays.
        t: Current timestep.

    Returns:
        The current epsilon value based on the linear decay.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class QPLEX:        
    """QPLEX implementation for training agents in a given environment: https://arxiv.org/abs/2008.01062
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):     
        """Init method for QPLEX

        Args:
            envs (gym.Env): The environments used for training.
            run_name (str): The name of the current training run.
            start_time (float): The time when training started.
            args (Dict[str, Any]): The command line arguments for configuration.
            ckpt (CheckpointSaver): The checkpoint handler for saving and loading training state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.train_freq % args.n_envs == 0, \
            f"Invalid train frequency: {args.train_freq}. Must be multiple of n_envs {args.n_envs}"

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # We don't take directly the env keys because agents are not ordered (0, 1, 2, ...) and causes problem with indexing elsewhere
        agent_ids = [f"agent_{idx}" for idx in range(len(envs.observation_space.keys()))]

        state_dim = sum(space.shape[0] for space in envs.observation_space.values()) if args.decentralized else envs.observation_space['agent_0'].shape[-1]

        # Initialize the Q-networks
        qnets, tg_qnets, buffers = {}, {}, {}
        for idx, agent in enumerate(agent_ids):
            qnets[agent] = QNetwork(idx, envs, args).to(device)
            
            tg_qnets[agent] = QNetwork(idx, envs, args).to(device)            
            tg_qnets[agent].load_state_dict(qnets[agent].state_dict())

            buffers[agent] = Buffer(agent, envs, args, state_dim, device)

        # Initialize the centralized mixer
        # n_agents, state_dim, act_dims, transf_embed_dim, n_heads, mix_embed_dim, mix_embed_layers, is_minus_one, weighted_head
        mixer_params = [len(agent_ids), state_dim, 
                            args.transf_embed_dim,
                            args.n_heads, args.mix_embed_dim,
                            args.mix_embed_layers, args.is_minus_one, args.weighted_head]
        mixer, tg_mixer = [QPLEXMixer(*mixer_params) for _ in range(2)]
        tg_mixer.load_state_dict(mixer.state_dict())

        if ckpt.resumed: 
            for agent in agent_ids:
                qnets[agent].load_state_dict(ckpt.loaded_run[agent])
                tg_qnets[agent].load_state_dict(qnets[agent].state_dict())
            mixer.load_state_dict(ckpt.loaded_run['mixer'])
            tg_mixer.load_state_dict(ckpt.loaded_run['mixer'])

        # Disable grad for tg nets to improve performance
        for agent in agent_ids:
            for param in tg_qnets[agent].parameters():
                param.requires_grad = False
        for param in tg_mixer.parameters():
            param.requires_grad = False

        # Initialize the optimizer and buffers
        qnet_params = [param for qnet in qnets.values() for param in qnet.parameters()]
        qnet_params += list(mixer.parameters())
        qnet_optim = optim.Adam(qnet_params, lr=args.lr)

        if ckpt.resumed: qnet_optim.load_state_dict(ckpt.loaded_run['qnet_optim'])

        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"
        logger = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        init_step = 1 if not ckpt.resumed else ckpt.loaded_run['last_step']
        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        obs, _ = envs.reset(seed=args.seed)
        obs = cast_np_to_tensors(obs, device)     

        try:
            for step in range(init_step, int(args.total_timesteps // args.n_envs)):
                global_step += args.n_envs
                epsilon = linear_schedule(
                    args.eps_start, args.eps_end, args.eps_decay_frac * args.total_timesteps, global_step
                )

                action = {}
                for agent in agent_ids:
                    if np.random.rand() < epsilon:
                            action[agent] = np.array([envs.action_space[agent].sample() for _ in range(envs.num_envs)])
                    else:
                        with th.no_grad():
                            action[agent] = qnets[agent].get_action(obs[agent]).cpu().numpy()

                next_obs, rewards, terminations, truncations, infos = envs.step(action)

                next_obs = cast_np_to_tensors(next_obs, device)      
                dones = np.logical_or(terminations[agent_ids[0]], truncations[agent_ids[0]])

                real_next_obs = next_obs.copy()                    
                for idx, done in enumerate(dones):
                    if done: 
                        for agent in agent_ids:
                            real_next_obs[agent][idx] = th.tensor(infos[idx]["final_observation"][agent]).to(device)
                    
                state = stack_agent_obs_by_env(obs) if args.decentralized else obs['agent_0']
                next_state = stack_agent_obs_by_env(real_next_obs) if args.decentralized else real_next_obs['agent_0']
                for agent in agent_ids:
                    buffers[agent].store(
                        obs[agent],
                        action[agent],
                        rewards[agent],
                        real_next_obs[agent],
                        done,
                        state,
                        next_state
                    )

                obs = next_obs.copy()           

                if global_step % args.eval_freq == 0:
                    evaluator.evaluate(global_step, qnets)
                    if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")

                if global_step > args.learning_starts:
                    if global_step % args.train_freq == 0:
                        q_vals, max_qvalue, actions, tg_q_vals, max_tg_q_vals, tg_actions = [[] for _ in range(6)]

                        for agent in agent_ids:
                            data = buffers[agent].sample()

                            with th.no_grad():
                                tg_action = qnets[agent](data['next_obs']).argmax(dim=1, keepdim=True)
                                tg_actions.append(tg_action)

                                tg_q_val = tg_qnets[agent](data['next_obs'])

                                tg_q_vals.append(tg_q_val.gather(-1, tg_action))
                                max_tg_q_vals.append(th.max(tg_q_val, dim=-1, keepdims=True)[0])

                            q_val = qnets[agent](data['obs'])
                            q_vals.append(q_val.gather(-1, data['action']))
                            max_qvalue.append(th.max(q_val, dim=-1, keepdims=True)[0])

                            actions.append(data['action'])

                        with th.no_grad():
                            reward = data['reward']
                            state = data['state']
                            
                            tg_q_vals = th.hstack(tg_q_vals)     # 128, 3

                            mix_tg_v_val = tg_mixer(tg_q_vals, data['next_state'], is_v=True).squeeze(-1)
                            mix_tg_a_val = tg_mixer(tg_q_vals, data["next_state"], actions=th.hstack(tg_actions),
                                max_q_i=th.hstack(max_tg_q_vals), is_v=False).squeeze(-1) 

                            mix_tg_q_val = mix_tg_v_val + mix_tg_a_val

                            mix_td_tg = reward + args.gamma * mix_tg_q_val * (1 - data['done'])
           
                        q_vals = th.hstack(q_vals)

                        ans_chosen = mixer(q_vals, data["state"], is_v=True).squeeze(-1)
                        ans_adv = mixer(q_vals, data['state'], 
                                        actions=th.hstack(actions),
                                        max_q_i=th.hstack(max_qvalue), is_v=False).squeeze(-1)
                        mix_q_val = ans_chosen + ans_adv  
                        
                        loss = F.mse_loss(mix_q_val, mix_td_tg)

                        # Optimize the model
                        qnet_optim.zero_grad()
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(qnet_params, 10)
                        qnet_optim.step()

                    # Update target network
                    if global_step % args.tg_qnet_freq == 0:
                        for agent in agent_ids:
                            for tg_qnet_param, qnet_param in zip(tg_qnets[agent].parameters(), qnets[agent].parameters()):
                                tg_qnet_param.data.copy_(
                                    args.tau * qnet_param.data + (1.0 - args.tau) * tg_qnet_param.data
                                )

                # If we reach the node's time limit, we just exit the training loop, save metrics and checkpoint
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            if args.checkpoint:
                # Save the checkpoint and logger data
                ckpt.set_record(args, qnets, mixer, global_step, qnet_optim, "" if not logger else logger.wb_path, step)
                ckpt.save()
            if logger: logger.close()
            envs.close()
