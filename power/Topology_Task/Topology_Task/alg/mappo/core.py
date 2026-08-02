from time import time

from .agent import Actor, Critic
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from common.utils import cast_np_to_tensors, stack_agent_obs_by_env
from env.eval import Evaluator

class MAPPO:
    """Multi-agent Proximal Policy Optimization (PPO) implementation for training an agent in a given environment: https://arxiv.org/abs/2103.01955.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        """Init method for PPO

        Args:
            envs (gym.Env): The environments used for training.
            run_name (str): The name of the current training run.
            start_time (float): The time when training started.
            args (Dict[str, Any]): The command line arguments for configuration.
            ckpt (CheckpointSaver): The checkpoint handler for saving and loading training state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.n_steps % args.n_envs == 0, \
            f"Invalid train frequency (n_steps): {args.n_steps}. Must be multiple of n_envs {args.n_envs}"

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # We don't take directly the env keys because agents are not ordered (0, 1, 2, ...) and causes problem with indexing elsewhere
        agent_ids = [f"agent_{idx}" for idx in range(len(envs.observation_space.keys()))]

        # Initialize the rollout, actor, critic, optimizer, and buffer
        batch_size = int(args.n_envs * args.n_steps)
        minibatch_size = int(batch_size // args.n_minibatches)
        n_rollouts = args.total_timesteps // batch_size
        init_rollout = 1 if not ckpt.resumed else ckpt.loaded_run['last_rollout']

        # Determine action space type
        continuous_actions = True if args.action_type == "redispatch" else False
        actors = {f"agent_{idx}": Actor(idx, envs, args, continuous_actions).to(device) 
                  for idx in range(len(agent_ids))}
    
        critic = Critic(envs, args)

        if ckpt.resumed:
            for agent in actors.keys():
                actors[agent].load_state_dict(ckpt.loaded_run[agent])
            critic.load_state_dict(ckpt.loaded_run['critic'])

        actor_params = [param for actor in actors.values() for param in actor.parameters()]
        if continuous_actions: raise("Redispatching actions are not yet implemented")
        actor_optim = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])

        joint_obs_size = sum(space.shape[0] for space in envs.observation_space.values()) if args.decentralized else envs.observation_space['agent_0'].shape[-1]
        joint_observations = th.zeros((args.n_steps, args.n_envs) + 
                                      (joint_obs_size,)).to(device)
        values = th.zeros((args.n_steps, args.n_envs)).to(device)
        dones = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        terminations = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        observations, actions, logprobs, rewards = [{} for _ in range(4)]
        for id in agent_ids:
            observations[id] = th.zeros((args.n_steps, args.n_envs) + envs.observation_space[id].shape).to(device)
            actions[id] = th.zeros((args.n_steps, args.n_envs)).to(device)   # Assuming discrete actions
            logprobs[id] = th.zeros((args.n_steps, args.n_envs)).to(device)
            rewards[id] = th.zeros((args.n_steps, args.n_envs)).to(device)
            
        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"
        logger = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        last_ckpt_time = start_time            # <-- track last checkpoint timestamp
        next_obs, _ = envs.reset()
        next_obs = cast_np_to_tensors(next_obs, device)     
        try:
            for iteration in range(init_rollout, n_rollouts + 1):
                # Annealing the rate if instructed to do so
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / n_rollouts
                    actor_optim.param_groups[0]['lr'] = frac * args.actor_lr
                    critic_optim.param_groups[0]['lr'] = frac * args.critic_lr

                for step in range(0, args.n_steps):
                    global_step += args.n_envs

                    action, logprob = {}, {}
                    for agent in agent_ids:
                        observations[agent][step] = next_obs[agent]
                    
                        with th.no_grad():
                            action[agent], logprob[agent], _ = actors[agent].get_action(next_obs[agent])

                        actions[agent][step] = action[agent]#.unsqueeze(-1)
                        logprobs[agent][step] = logprob[agent]#.unsqueeze(-1)

                    # get joint obs for this
                    with th.no_grad():   
                        joint_obs = stack_agent_obs_by_env(next_obs) if args.decentralized else next_obs['agent_0']
                        value = critic.get_value(joint_obs)
                        joint_observations[step] = joint_obs
                        values[step] = value.flatten()
                        
                    next_obs, reward, next_terminations, next_truncations, infos = envs.step(action)
                
                    reward = cast_np_to_tensors(reward, device)
                    for agent in agent_ids: rewards[agent][step] = reward[agent]

                    dones[step] = th.tensor(
                        np.logical_or(next_terminations[agent_ids[0]], next_truncations[agent_ids[0]])
                    ).to(device)
                    terminations[step] = th.tensor(next_terminations[agent_ids[0]]).to(device)

                    next_obs = cast_np_to_tensors(next_obs, device)      
                    real_next_obs = next_obs.copy()                    
                    for idx, done in enumerate(dones[step]):
                        if done: 
                            for agent in agent_ids:
                                real_next_obs[agent][idx] = th.tensor(infos[idx]["final_observation"][agent]).to(device)
                    
                    if global_step % args.eval_freq == 0:
                        evaluator.evaluate(global_step, actors)
                        if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")

                # Bootstrap value if not done
                with th.no_grad():
                    advantages, returns = {}, {}
                    joint_real_next_obs = stack_agent_obs_by_env(real_next_obs) if args.decentralized else real_next_obs['agent_0']
                    for agent in agent_ids:
                        advantages[agent] = th.zeros_like(rewards[agent]).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.n_steps)):
                            if t == args.n_steps - 1:
                                nextvalues = critic.get_value(joint_real_next_obs).reshape(1, -1)
                            else:
                                nextvalues = values[t + 1]
                            delta = rewards[agent][t] + args.gamma * nextvalues * (1 - terminations[t]) - values[t]
                            advantages[agent][t] = lastgaelam = delta + args.gamma * args.gae_lambda * (1 - dones[t]) * lastgaelam
                        returns[agent] = advantages[agent] + values
    
                # --- HOURLY CHECKPOINT (wall clock) ---
                if time() - last_ckpt_time >= 3600:
                    ckpt.set_record(
                        args, actors, critic, global_step,
                        actor_optim, critic_optim,
                        "" if not logger else logger.wb_path,
                        iteration
                    )
                    ckpt.save()            # overwrites the previous checkpoint
                    last_ckpt_time = time()
                # --------------------------------------
            
                b_values = values.reshape(-1)
                b_joint_obs = joint_observations.reshape((-1,) + (joint_obs_size,))
                for agent in agent_ids:
                    # Flatten the batch
                    b_obs = observations[agent].reshape((-1,) + envs.observation_space[agent].shape)
                    b_logprobs = logprobs[agent].reshape(-1)
                    b_actions = actions[agent].reshape(-1,)
                    b_advantages = advantages[agent].reshape(-1)
                    b_returns = returns[agent].reshape(-1)

                    # Optimizing the policy and value network
                    b_inds = np.arange(batch_size)
                    clipfracs = []
                    for _ in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, batch_size, minibatch_size):
                            end = start + minibatch_size
                            mb_inds = b_inds[start:end]
                            action, newlogprob, entropy = actors[agent].get_action(b_obs[mb_inds], b_actions.long()[mb_inds])
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with th.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                #old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss = th.max(pg_loss1, pg_loss2).mean()

                            entropy_loss = entropy.mean()
                            pg_loss = pg_loss - args.entropy_coef * entropy_loss

                            actor_optim.zero_grad()
                            pg_loss.backward()
                            nn.utils.clip_grad_norm_(actors[agent].parameters(), args.max_grad_norm)
                            actor_optim.step()

                            # Value loss
                            newvalue = critic.get_value(b_joint_obs[mb_inds]).view(-1)
                            if args.clip_vfloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + th.clamp(
                                    newvalue - b_values[mb_inds],
                                    -args.clip_coef,
                                    args.clip_coef,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                            v_loss *= args.vf_coef

                            critic_optim.zero_grad()
                            v_loss.backward()
                            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                            critic_optim.step()

                        if args.target_kl is not None and approx_kl > args.target_kl:
                            break

                # If we reach the node's time limit, we just exit the training loop, save metrics and ckpt
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            if args.checkpoint:
                # Save the checkpoint and logger data
                ckpt.set_record(args, actors, critic, global_step, actor_optim, critic_optim, "" if not logger else logger.wb_path, iteration)
                ckpt.save() 
            if logger: logger.close()
            envs.close()
