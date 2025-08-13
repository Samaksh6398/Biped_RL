# ppo_lstm_policy.py (Corrected)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# ... (ActorCriticLSTM class is unchanged) ...
class ActorCriticLSTM(nn.Module):
    """
    PPO-LSTM Actor-Critic Network
    
    This combines the policy (actor) and value function (critic) with LSTM memory.
    The LSTM allows the network to remember past observations and make better decisions.
    """
    
    def __init__(
        self,
        num_obs,           # Number of observations (38 for your biped)
        num_actions,       # Number of actions (9 for your biped)
        actor_hidden_dims=[512, 256],  # Hidden layer dimensions for actor
        critic_hidden_dims=[512, 256], # Hidden layer dimensions for critic  
        lstm_hidden_size=256,          # LSTM hidden state size
        lstm_num_layers=1,             # Number of LSTM layers
        activation='elu',              # Activation function
        init_noise_std=1.0,           # Initial noise for exploration
        **kwargs
    ):
        super(ActorCriticLSTM, self).__init__()
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Activation function
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 1. OBSERVATION ENCODER
        # This processes raw observations before feeding to LSTM
        encoder_layers = []
        input_dim = num_obs
        
        # Add encoding layers to compress observations
        encoder_dims = [256, 128]  # Compress to smaller representation for LSTM
        for hidden_dim in encoder_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(self.activation)
            input_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        lstm_input_dim = input_dim  # 128 from encoder output
        
        # 2. LSTM LAYER
        # This is the memory component that remembers past states
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True  # Input shape: (batch, seq_len, features)
        )
        
        # 3. ACTOR NETWORK (Policy)
        # Takes LSTM output and produces action distribution
        actor_layers = []
        input_dim = lstm_hidden_size
        
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(self.activation)
            input_dim = hidden_dim
            
        self.actor_backbone = nn.Sequential(*actor_layers)
        
        # Policy head - outputs mean of action distribution
        self.policy_head = nn.Linear(input_dim, num_actions)
        
        # Log standard deviation for action noise (learnable parameter)
        self.log_std = nn.Parameter(
            torch.ones(num_actions) * np.log(init_noise_std)
        )
        
        # 4. CRITIC NETWORK (Value Function)
        # Takes LSTM output and estimates state value
        critic_layers = []
        input_dim = lstm_hidden_size
        
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(self.activation)
            input_dim = hidden_dim
            
        self.critic_backbone = nn.Sequential(*critic_layers)
        
        # Value head - outputs single value estimate
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                # Special initialization for LSTM
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def init_hidden(self, batch_size, device):
        """
        Initialize LSTM hidden states
        
        Returns:
            Tuple of (hidden_state, cell_state) for LSTM
            Each has shape: (num_layers, batch_size, hidden_size)
        """
        hidden = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=torch.float32
        )
        cell = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=torch.float32
        )
        return (hidden, cell)
    
    def forward(self, observations, lstm_states=None, masks=None):
        """
        Forward pass through the network
        
        Args:
            observations: Current observations [batch_size, num_obs]
            lstm_states: Previous LSTM states (hidden, cell) or None
            masks: Reset mask for episodes [batch_size, 1] (1=continue, 0=reset)
            
        Returns:
            actions: Sampled actions [batch_size, num_actions]  
            action_log_probs: Log probabilities of actions [batch_size, 1]
            values: State value estimates [batch_size, 1]
            new_lstm_states: Updated LSTM states
            entropy: Action distribution entropy [batch_size, 1]
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # 1. ENCODE OBSERVATIONS
        encoded_obs = self.encoder(observations)
        
        # Reshape for LSTM: [batch_size, seq_len=1, features]
        encoded_obs = encoded_obs.unsqueeze(1)
        
        # 2. HANDLE LSTM STATES
        if lstm_states is None:
            # Initialize new hidden states if not provided
            lstm_states = self.init_hidden(batch_size, device)
        
        hidden, cell = lstm_states
        
        # Apply episode masks to reset LSTM states when episodes end
        if masks is not None:
            # Ensure masks has correct shape: [batch_size, 1]
            if masks.dim() == 1:
                masks = masks.unsqueeze(-1)
            # Expand masks to match LSTM hidden state dimensions: [num_layers, batch_size, hidden_size]
            # masks: [batch_size, 1] -> [batch_size, hidden_size] -> [num_layers, batch_size, hidden_size]
            mask_expanded = masks.expand(batch_size, self.lstm_hidden_size).unsqueeze(0).expand_as(hidden)
            hidden = hidden * mask_expanded
            cell = cell * mask_expanded
        
        # 3. LSTM FORWARD PASS
        lstm_out, new_lstm_states = self.lstm(encoded_obs, (hidden, cell))
        
        # Remove sequence dimension: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        lstm_out = lstm_out.squeeze(1)
        
        # 4. ACTOR (POLICY) FORWARD
        actor_features = self.actor_backbone(lstm_out)
        action_mean = self.policy_head(actor_features)
        
        # Create action distribution (diagonal Gaussian)
        action_std = self.log_std.exp().expand_as(action_mean)
        action_dist = Normal(action_mean, action_std)
        
        # Sample actions
        actions = action_dist.sample()
        
        # Compute log probabilities
        action_log_probs = action_dist.log_prob(actions).sum(dim=1, keepdim=True)
        
        # Compute entropy for exploration bonus
        entropy = action_dist.entropy().sum(dim=1, keepdim=True)
        
        # 5. CRITIC (VALUE) FORWARD
        critic_features = self.critic_backbone(lstm_out)
        values = self.value_head(critic_features)
        
        return actions, action_log_probs, values, new_lstm_states, entropy
    
    def act(self, observations, lstm_states=None, masks=None, deterministic=False):
        """
        Action selection for evaluation/inference
        
        Args:
            observations: Current observations
            lstm_states: Previous LSTM states  
            masks: Episode reset masks
            deterministic: If True, use mean action (no noise)
            
        Returns:
            actions: Selected actions
            new_lstm_states: Updated LSTM states
        """
        with torch.no_grad():
            batch_size = observations.shape[0]
            device = observations.device
            
            # Encode observations
            encoded_obs = self.encoder(observations).unsqueeze(1)
            
            # Handle LSTM states
            if lstm_states is None:
                lstm_states = self.init_hidden(batch_size, device)
                
            hidden, cell = lstm_states
            
            # Apply masks
            if masks is not None:
                # Ensure masks has correct shape: [batch_size, 1]
                if masks.dim() == 1:
                    masks = masks.unsqueeze(-1)
                # Expand masks to match LSTM hidden state dimensions
                mask_expanded = masks.expand(batch_size, self.lstm_hidden_size).unsqueeze(0).expand_as(hidden)
                hidden = hidden * mask_expanded
                cell = cell * mask_expanded
            
            # LSTM forward
            lstm_out, new_lstm_states = self.lstm(encoded_obs, (hidden, cell))
            lstm_out = lstm_out.squeeze(1)
            
            # Get action mean
            actor_features = self.actor_backbone(lstm_out)
            action_mean = self.policy_head(actor_features)
            
            if deterministic:
                actions = action_mean
            else:
                # Sample from distribution
                action_std = self.log_std.exp().expand_as(action_mean)
                action_dist = Normal(action_mean, action_std)
                actions = action_dist.sample()
            
            return actions, new_lstm_states
    
    def evaluate_actions(self, observations, actions, lstm_states, masks):
        """
        Evaluate actions for PPO training
        
        This is used during training to compute policy gradients.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions to evaluate  
            lstm_states: LSTM states for the batch
            masks: Episode masks
            
        Returns:
            action_log_probs: Log probabilities of given actions
            values: State value estimates
            entropy: Policy entropy
        """
        batch_size = observations.shape[0]
        
        # Encode observations
        encoded_obs = self.encoder(observations).unsqueeze(1)
        
        # Handle LSTM states
        hidden, cell = lstm_states
        if masks is not None:
            # Ensure masks has correct shape: [batch_size, 1]
            if masks.dim() == 1:
                masks = masks.unsqueeze(-1)
            # Expand masks to match LSTM hidden state dimensions
            mask_expanded = masks.expand(batch_size, self.lstm_hidden_size).unsqueeze(0).expand_as(hidden)
            hidden = hidden * mask_expanded
            cell = cell * mask_expanded
        
        # LSTM forward
        lstm_out, _ = self.lstm(encoded_obs, (hidden, cell))
        lstm_out = lstm_out.squeeze(1)
        
        # Actor evaluation
        actor_features = self.actor_backbone(lstm_out)
        action_mean = self.policy_head(actor_features)
        
        action_std = self.log_std.exp().expand_as(action_mean)
        action_dist = Normal(action_mean, action_std)
        
        action_log_probs = action_dist.log_prob(actions).sum(dim=1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=1, keepdim=True)
        
        # Critic evaluation
        critic_features = self.critic_backbone(lstm_out)
        values = self.value_head(critic_features)
        
        return action_log_probs, values, entropy


class PPOLSTMAgent:
    """
    PPO-LSTM Training Agent
    
    This handles the training loop and rollout collection with LSTM states.
    """
    
    def __init__(
        self,
        policy,
        device,
        num_envs,
        num_steps_per_env,
        mini_batch_size,
        num_learning_epochs,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        lr=3e-4,
        max_grad_norm=1.0,
        **kwargs
    ):
        self.policy = policy
        self.device = device
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.mini_batch_size = mini_batch_size
        self.num_learning_epochs = num_learning_epochs
        
        # PPO hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for rollouts
        self.storage = RolloutStorage(
            num_steps_per_env,
            num_envs, 
            policy.num_obs,
            policy.num_actions,
            policy.lstm_hidden_size,
            policy.lstm_num_layers,
            device
        )
        
    def update(self):
        """
        Update policy using collected rollouts.
        
        Returns:
            dict: A dictionary containing the mean losses for logging.
        """
        
        # Compute GAE advantages
        self.storage.compute_returns_and_advantages(
            self.policy, self.gamma, self.lam
        )
        
        # Get training data
        obs_batch, actions_batch, values_batch, returns_batch, \
        advantages_batch, old_log_probs_batch, lstm_states_batch, masks_batch = \
            self.storage.get_training_data()
        
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / \
                          (advantages_batch.std() + 1e-8)
        
        # --- START OF FIX: Initialize loss trackers ---
        policy_loss_list = []
        value_loss_list = []
        entropy_loss_list = []
        approx_kl_list = []
        # --- END OF FIX ---

        # Training loop
        for epoch in range(self.num_learning_epochs):
            
            batch_size = self.num_envs * self.num_steps_per_env
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch data
                obs_mb = obs_batch[batch_indices]
                actions_mb = actions_batch[batch_indices]  
                values_mb = values_batch[batch_indices]
                returns_mb = returns_batch[batch_indices]
                advantages_mb = advantages_batch[batch_indices]
                old_log_probs_mb = old_log_probs_batch[batch_indices]
                
                # LSTM states need special handling for mini-batches
                # Slicing along the batch dimension (dim=1)
                lstm_hidden_mb = lstm_states_batch[0][:, batch_indices]
                lstm_cell_mb = lstm_states_batch[1][:, batch_indices]
                lstm_states_mb = (lstm_hidden_mb, lstm_cell_mb)
                
                masks_mb = masks_batch[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, values_pred, entropy = self.policy.evaluate_actions(
                    obs_mb, actions_mb, lstm_states_mb, masks_mb
                )
                
                # PPO loss computation
                ratio = torch.exp(log_probs - old_log_probs_mb)
                
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values_pred, returns_mb)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + \
                           self.value_loss_coef * value_loss + \
                           self.entropy_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # --- START OF FIX: Append losses for logging ---
                policy_loss_list.append(policy_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                # Calculate approximate KL divergence
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs_mb
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_list.append(approx_kl)
                # --- END OF FIX ---
        
        # Clear storage for next rollout
        self.storage.clear()
        
        # --- START OF FIX: Return dictionary of mean losses ---
        mean_losses = {
            'policy_loss': np.mean(policy_loss_list),
            'value_loss': np.mean(value_loss_list),
            'entropy_loss': np.mean(entropy_loss_list),
            'approx_kl': np.mean(approx_kl_list),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        return mean_losses
        # --- END OF FIX ---

# ... (RolloutStorage class is unchanged) ...
class RolloutStorage:
    """
    Storage for PPO-LSTM rollouts
    
    This stores experiences during rollout collection, including LSTM states.
    """
    
    def __init__(self, num_steps, num_envs, obs_dim, action_dim, 
                 lstm_hidden_size, lstm_num_layers, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # Storage tensors
        self.observations = torch.zeros(num_steps + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=device)
        self.values = torch.zeros(num_steps + 1, num_envs, 1, device=device)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, 1, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=device)
        self.masks = torch.ones(num_steps + 1, num_envs, 1, device=device)
        
        # LSTM states storage
        self.lstm_hidden_states = torch.zeros(
            num_steps + 1, lstm_num_layers, num_envs, lstm_hidden_size, device=device
        )
        self.lstm_cell_states = torch.zeros(
            num_steps + 1, lstm_num_layers, num_envs, lstm_hidden_size, device=device  
        )
        
        self.step = 0
    
    def insert(self, obs, actions, rewards, values, log_probs, lstm_states, masks):
        """Insert a step of experience"""
        self.observations[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))  # Ensure rewards have correct shape
        self.values[self.step].copy_(values)
        self.log_probs[self.step].copy_(log_probs)
        self.masks[self.step + 1].copy_(masks.view(-1, 1))  # Ensure masks have correct shape
        
        # Store LSTM states
        hidden, cell = lstm_states
        self.lstm_hidden_states[self.step + 1].copy_(hidden)
        self.lstm_cell_states[self.step + 1].copy_(cell)
        
        self.step += 1
    
    def compute_returns_and_advantages(self, policy, gamma, lam):
        """Compute GAE advantages and returns"""
        with torch.no_grad():
            # Get value of last state
            last_obs = self.observations[-1]
            last_lstm_states = (
                self.lstm_hidden_states[-1],
                self.lstm_cell_states[-1] 
            )
            last_masks = self.masks[-1]
            
            # Dummy forward pass to get last value
            _, _, last_values, _, _ = policy(last_obs, last_lstm_states, last_masks)
            self.values[-1] = last_values
            
            # Compute GAE
            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
                gae = delta + gamma * lam * self.masks[step + 1] * gae
                self.advantages[step] = gae
                self.returns[step] = gae + self.values[step]
    
    def get_training_data(self):
        """Get flattened training data"""
        # Flatten data for training
        batch_size = self.num_steps * self.num_envs
        
        obs = self.observations[:-1].view(batch_size, -1)
        actions = self.actions.view(batch_size, -1)
        values = self.values[:-1].view(batch_size, -1)
        returns = self.returns[:-1].view(batch_size, -1)
        advantages = self.advantages.view(batch_size, -1)
        log_probs = self.log_probs.view(batch_size, -1)
        masks = self.masks[:-1].view(batch_size, -1)
        
        # LSTM states - keep layer dimension
        # Reshape to (num_layers, batch_size, hidden_size)
        lstm_hidden = self.lstm_hidden_states[0].permute(1, 0, 2).reshape(self.num_envs, -1)
        lstm_hidden = self.lstm_hidden_states[:-1].transpose(0, 2).reshape(self.num_envs * self.num_steps, self.lstm_hidden_states.shape[1], -1).transpose(0,1)
        lstm_cell = self.lstm_cell_states[:-1].transpose(0, 2).reshape(self.num_envs * self.num_steps, self.lstm_cell_states.shape[1], -1).transpose(0,1)

        lstm_states = (lstm_hidden, lstm_cell)
        
        return obs, actions, values, returns, advantages, log_probs, lstm_states, masks
    
    def clear(self):
        """Clear storage and reset step counter"""
        self.step = 0
        # Copy last observation and LSTM states to first position for next rollout
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
        self.lstm_hidden_states[0].copy_(self.lstm_hidden_states[-1])
        self.lstm_cell_states[0].copy_(self.lstm_cell_states[-1])