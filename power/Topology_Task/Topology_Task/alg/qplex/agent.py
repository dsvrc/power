from common.imports import *
from common.utils import Linear, th_act_fns

class QNetwork(nn.Module):
    """Q-Network for a reinforcement learning agent.

    This network predicts Q-values for given states, allowing the agent to select actions.

    Attributes:
        qnet (nn.Sequential): Sequential neural network model for Q-value prediction.
    """

    def __init__(self, id: int, envs: gym.Env, args: Dict[str, Any]):
        """Initialize the Q-Network.

        Args:
            envs: Environment(s) with defined observation and action spaces.
            args: Arguments containing network configuration, including activation function and layer sizes.
        """
        super().__init__()

        act_str, act_fn = args.act_fn, th_act_fns[args.act_fn]

        layers = []
        layers.extend([
            Linear(np.prod(envs.observation_space[f"agent_{id}"].shape), args.layers[0], act_str), 
            act_fn
        ])
        for idx, embed_dim in enumerate(args.layers[1:], start=1): 
            layers.extend([Linear(args.layers[idx-1], embed_dim, act_str), act_fn])
        
        layers.append(Linear(args.layers[-1], envs.action_space[f"agent_{id}"].n, 'linear'))

        self.qnet = nn.Sequential(*layers)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass through the Q-Network.

        Args:
            x: Input tensor representing the state.

        Returns:
            Tensor with Q-values for each action.
        """
        return self.qnet(x)

    def get_action(self, x: th.Tensor) -> np.ndarray:
        """Get the action with the highest Q-value.

        Args:
            x: Input tensor representing the state.

        Returns:
            Numpy array of selected actions.
        """
        q_values = self(x)
        actions = th.argmax(q_values, dim=1)
        return actions
    
    def get_eval_action(self, x: th.Tensor) -> np.ndarray:
        """Get the action for evaluation.

        Args:
            x: Input tensor representing the state.

        Returns:
            Numpy array of selected actions for evaluation.
        """
        q_values = self(x)
        actions = th.argmax(q_values)
        return actions


class QPLEXMixer(nn.Module):
    def __init__(self, n_agents, state_dim, transf_embed_dim, n_heads, mix_embed_dim, 
                 mix_embed_layers, is_minus_one, weighted_head, detach=0):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.detach = detach

        self.action_dim = n_agents

        self.mlp_w = nn.Sequential(nn.Linear(self.state_dim, transf_embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(transf_embed_dim, self.n_agents))
        self.v_bias = nn.Sequential(nn.Linear(self.state_dim, transf_embed_dim),
                               nn.ReLU(),
                               nn.Linear(transf_embed_dim, self.n_agents))

        self.si_weight = AttentionQPLEX(n_agents, self.state_dim, self.action_dim,
                                         n_heads, mix_embed_dim, mix_embed_layers)

        self.is_minus_one = is_minus_one
        self.weighted_head = weighted_head
    
    def compute_v_mix(self, q_vals):
        return th.sum(q_vals.view(-1, self.n_agents), dim=-1)

    def compute_a_mix(self, q_vals, states, actions, max_q_i):
        a_vals = (q_vals - max_q_i).view(-1, self.n_agents).detach()
        
        # Weights for the advantage stream
        a_w = self.si_weight(states, actions)
        a_w = a_w.view(-1, self.n_agents)

        # Compute mixed advantage
        if self.is_minus_one: 
            return th.sum(a_vals * (a_w - 1.), dim=1)
        return th.sum(a_vals * a_w, dim=1)

    def calc(self, q_vals, states, actions=None, max_q_i=None, is_v=False):
        # Compute mixed state value else mixed advantage
        if is_v:
            return self.compute_v_mix(q_vals)

        return self.compute_a_mix(q_vals, states, actions, max_q_i)

    # max_q_i is the advantage of agent's i selected action
    def forward(self, 
                q_vals,     # q_val for agents' picked action
                states, 
                actions=None,   # one-hot encoding of the picked action
                max_q_i=None,   # max_q_val according to current net
                is_v=False      # whether we are combining V or A
        ):
        bs = q_vals.size(0)
        states = states.reshape(-1, self.state_dim)
        q_vals = q_vals.view(-1, self.n_agents)
      
        w_2 = th.abs(self.mlp_w(states)).view(-1, self.n_agents) + 1e-10
        
        v = self.v_bias(states)
        v = v.view(-1, self.n_agents)

        if self.weighted_head:
            q_vals = w_2 * q_vals + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)   # We already flattened this in the loss
            if self.weighted_head:
                max_q_i = w_2 * max_q_i + v
        mixed_value = self.calc(q_vals, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        return mixed_value.view(bs, -1, 1)
    
class AttentionQPLEX(nn.Module):
    def __init__(self, n_agents, state_dim, action_dim, n_heads, mix_embed_dim, mix_embed_layers):
        super().__init__()

        self.n_agents = n_agents
        state_action_dim = state_dim + action_dim

        self.n_heads = n_heads
        self.key_extractors, self.agents_extractors, self.action_extractors = [nn.ModuleList() for _ in range(3)]
    
        for _ in range(self.n_heads):  # multi-head attention
            if mix_embed_layers == 1:
                self.key_extractors.append(nn.Linear(state_dim, 1))  # key
                self.agents_extractors.append(nn.Linear(state_dim, n_agents))  # agent
                self.action_extractors.append(nn.Linear(state_action_dim, n_agents))  # action
            elif mix_embed_layers == 2:
                self.key_extractors.append(nn.Sequential(nn.Linear(state_dim, mix_embed_dim),
                                                         nn.ReLU(),
                                                         nn.Linear(mix_embed_dim, 1)))  # key
                self.agents_extractors.append(nn.Sequential(nn.Linear(state_dim, mix_embed_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(mix_embed_dim, n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(state_action_dim, mix_embed_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(mix_embed_dim, n_agents)))  # action
            elif mix_embed_layers == 3:
                self.key_extractors.append(nn.Sequential(nn.Linear(state_dim,mix_embed_dim),
                                                         nn.ReLU(),
                                                         nn.Linear(mix_embed_dim, mix_embed_dim),
                                                         nn.ReLU(),
                                                         nn.Linear(mix_embed_dim, 1)))  # key
                self.agents_extractors.append(nn.Sequential(nn.Linear(state_dim, mix_embed_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(mix_embed_dim, mix_embed_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(mix_embed_dim, n_agents)))  # agent
                self.action_extractors.append(nn.Sequential(nn.Linear(state_action_dim, mix_embed_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(mix_embed_dim, mix_embed_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(mix_embed_dim, n_agents)))  # action
            else:
                raise Exception("Error setting number of adv hypernet layers.")

    def forward(self, states, actions):
        # From (batch_size, n_agents, act_dim) to (batch_size, n_agents * act_dim)  
        actions = actions.reshape(actions.shape[0], -1)  
        data = th.cat([states, actions], dim=1)

        heads_key = [k_ext(states) for k_ext in self.key_extractors]
        heads_agents = [k_ext(states) for k_ext in self.agents_extractors]
        heads_actions = [sel_ext(data) for sel_ext in self.action_extractors]

        attention_w = []
        for h_key, h_agents, h_actions in zip(heads_key, heads_agents, heads_actions):
            w_key = th.abs(h_key).repeat(1, self.n_agents) + 1e-10
            w_agents = F.sigmoid(h_agents)
            w_actions = F.sigmoid(h_actions)
            weights = w_key * w_agents * w_actions
            attention_w.append(weights)
 
        attention_w = th.stack(attention_w, dim=1)
        attention_w = attention_w.view(-1, self.n_heads, self.n_agents)
        attention_w = th.sum(attention_w, dim=1)

        return attention_w