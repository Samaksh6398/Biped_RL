# biped_gym_wrapper.py
# Gymnasium wrapper for BipedEnv to make it compatible with Stable Baselines3

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple


class BipedGymWrapper(gym.Env):
    """
    Gymnasium wrapper for BipedEnv to make it compatible with Stable Baselines3.
    
    This wrapper handles the conversion between the custom BipedEnv interface
    and the standard Gymnasium interface expected by SB3.
    """
    
    def __init__(self, biped_env):
        """
        Initialize the wrapper.
        
        Args:
            biped_env: Instance of BipedEnv
        """
        super().__init__()
        
        self.biped_env = biped_env
        self.num_envs = biped_env.num_envs
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(biped_env.num_actions,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(biped_env.num_obs,),
            dtype=np.float32
        )
        
        # For vectorized environments
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        
        # Track current environment index for vectorized envs
        self.current_env_idx = 0
        
        # Store the last observations and infos
        self._last_obs = None
        self._last_infos = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        obs, extras = self.biped_env.reset()
        
        # Convert torch tensor to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
            
        # For vectorized environment, return the observation for the current environment
        if len(obs.shape) > 1:
            obs = obs[self.current_env_idx]
            
        info = self._process_extras(extras)
        
        self._last_obs = obs
        self._last_infos = info
        
        return obs.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward for the action
            terminated: Whether the episode terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Convert numpy action to torch tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.biped_env.device)
            
        # For vectorized environment, we need to handle single actions properly
        if len(action.shape) == 1:
            # Single action, expand to match num_envs
            action = action.unsqueeze(0).repeat(self.num_envs, 1)
            
        obs, rewards, dones, extras = self.biped_env.step(action)
        
        # Convert torch tensors to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
            
        # For single environment, extract the values for current env
        if len(obs.shape) > 1:
            obs = obs[self.current_env_idx]
        if len(rewards.shape) > 0:
            reward = float(rewards[self.current_env_idx])
        else:
            reward = float(rewards)
        if len(dones.shape) > 0:
            terminated = bool(dones[self.current_env_idx])
        else:
            terminated = bool(dones)
            
        # Handle truncation (not used in this environment)
        truncated = False
        
        info = self._process_extras(extras)
        
        # Store for potential access
        self._last_obs = obs
        self._last_infos = info
        
        return obs.astype(np.float32), reward, terminated, truncated, info
    
    def _process_extras(self, extras: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process extras dictionary from BipedEnv to create info dict for Gymnasium.
        
        Args:
            extras: Extras dictionary from BipedEnv
            
        Returns:
            info: Processed info dictionary
        """
        info = {}
        
        if extras is None:
            return info
            
        # Extract episode information if available
        if "episode" in extras:
            for key, value in extras["episode"].items():
                if isinstance(value, torch.Tensor):
                    info[key] = value.cpu().numpy()
                else:
                    info[key] = value
                    
        # Extract other useful information
        for key in ["fps", "time_outs"]:
            if key in extras:
                value = extras[key]
                if isinstance(value, torch.Tensor):
                    if len(value.shape) > 0:
                        info[key] = value[self.current_env_idx].cpu().numpy()
                    else:
                        info[key] = value.cpu().numpy()
                else:
                    info[key] = value
                    
        return info
    
    def render(self, mode: str = "human"):
        """
        Render the environment.
        
        Args:
            mode: Render mode
        """
        # BipedEnv uses Genesis for rendering, which is handled internally
        # This is a placeholder for compatibility
        pass
    
    def close(self):
        """Close the environment."""
        # BipedEnv doesn't need explicit closing
        pass
    
    def seed(self, seed: int):
        """Set the random seed."""
        np.random.seed(seed)
        torch.manual_seed(seed)


class VectorizedBipedWrapper(gym.vector.VectorEnv):
    """
    Vectorized wrapper for BipedEnv that properly handles multiple environments.
    """
    
    def __init__(self, biped_env):
        """
        Initialize the vectorized wrapper.
        
        Args:
            biped_env: Instance of BipedEnv
        """
        self.biped_env = biped_env
        
        # Define action and observation spaces
        single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(biped_env.num_actions,),
            dtype=np.float32
        )
        
        single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(biped_env.num_obs,),
            dtype=np.float32
        )
        
        super().__init__(
            num_envs=biped_env.num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        obs, extras = self.biped_env.reset()
        
        # Convert torch tensor to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
            
        infos = self._process_vectorized_extras(extras)
        
        return obs.astype(np.float32), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Step all environments."""
        # Convert numpy actions to torch tensor
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(self.biped_env.device)
            
        obs, rewards, dones, extras = self.biped_env.step(actions)
        
        # Convert torch tensors to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
            
        # Handle truncation (not used in this environment)
        truncated = np.zeros_like(dones, dtype=bool)
        
        infos = self._process_vectorized_extras(extras)
        
        return obs.astype(np.float32), rewards.astype(np.float32), dones, truncated, infos
    
    def _process_vectorized_extras(self, extras: Dict[str, Any]) -> list:
        """Process extras for vectorized environments."""
        infos = [{} for _ in range(self.num_envs)]
        
        if extras is None:
            return infos
            
        # Extract episode information if available
        if "episode" in extras:
            for i in range(self.num_envs):
                for key, value in extras["episode"].items():
                    if isinstance(value, torch.Tensor):
                        infos[i][key] = value.cpu().numpy()
                    else:
                        infos[i][key] = value
                        
        # Extract other useful information
        for key in ["fps", "time_outs"]:
            if key in extras:
                value = extras[key]
                if isinstance(value, torch.Tensor):
                    if len(value.shape) > 0:
                        for i in range(self.num_envs):
                            infos[i][key] = value[i].cpu().numpy()
                    else:
                        for i in range(self.num_envs):
                            infos[i][key] = value.cpu().numpy()
                else:
                    for i in range(self.num_envs):
                        infos[i][key] = value
                        
        return infos
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        pass
    
    def close(self):
        """Close the environment."""
        pass
    
    def seed(self, seed: int):
        """Set the random seed."""
        np.random.seed(seed)
        torch.manual_seed(seed)


def make_biped_env(env_cfg, obs_cfg, reward_cfg, command_cfg, num_envs=1, **kwargs):
    """
    Factory function to create a wrapped BipedEnv compatible with SB3.
    
    Args:
        env_cfg: Environment configuration
        obs_cfg: Observation configuration  
        reward_cfg: Reward configuration
        command_cfg: Command configuration
        num_envs: Number of environments
        **kwargs: Additional arguments
        
    Returns:
        Wrapped environment compatible with SB3
    """
    from biped_env import BipedEnv
    
    # Create the base environment
    biped_env = BipedEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=kwargs.get('show_viewer', False)
    )
    
    if num_envs == 1:
        # Single environment wrapper
        return BipedGymWrapper(biped_env)
    else:
        # Vectorized environment wrapper
        return VectorizedBipedWrapper(biped_env)