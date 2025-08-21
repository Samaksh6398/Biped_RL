import numpy as np
import wandb
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class WandbRewardCallback(BaseCallback):
    """
    A custom callback to log episode rewards and mean rewards to Weights & Biases.
    This callback monitors the info dict from VecEnv for episode statistics.
    """
    
    def __init__(self, log_freq=1000, mean_window=100, verbose=0):
        """
        Args:
            log_freq: Frequency (in timesteps) to log statistics to wandb
            mean_window: Window size for calculating mean rewards
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.mean_window = mean_window
        
        # Track episode rewards
        self.episode_rewards = deque(maxlen=mean_window)
        self.episode_lengths = deque(maxlen=mean_window)
        
        # Statistics
        self.total_episodes = 0
        self.last_log_step = 0
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if infos contain episode data
        if 'infos' in self.locals:
            infos = self.locals['infos']
            
            # Iterate through all environment infos
            for info in infos:
                # SB3 VecEnv puts episode info in a special 'episode' key when episodes end
                if 'episode' in info:
                    episode_info = info['episode']
                    
                    # Extract episode reward and length
                    episode_reward = episode_info['r']  # SB3 standard key for episode reward
                    episode_length = episode_info['l']  # SB3 standard key for episode length
                    
                    # Store episode statistics
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.total_episodes += 1
                    
                    # Log individual episode reward to wandb immediately
                    wandb.log({
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/total_episodes": self.total_episodes,
                    }, step=self.num_timesteps)
                    
                    if self.verbose > 0:
                        print(f"Episode {self.total_episodes} completed: reward={episode_reward:.3f}, length={episode_length}")
        
        # Log aggregate statistics periodically
        if (self.num_timesteps - self.last_log_step) >= self.log_freq and len(self.episode_rewards) > 0:
            self._log_statistics()
            self.last_log_step = self.num_timesteps
        
        return True
    
    def _log_statistics(self):
        """Log aggregate reward statistics to wandb."""
        if len(self.episode_rewards) == 0:
            return
            
        rewards_array = np.array(self.episode_rewards)
        lengths_array = np.array(self.episode_lengths)
        
        # Calculate statistics
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)
        min_reward = np.min(rewards_array)
        max_reward = np.max(rewards_array)
        
        mean_length = np.mean(lengths_array)
        std_length = np.std(lengths_array)
        
        # Log to wandb
        wandb.log({
            "rewards/mean_reward": mean_reward,
            "rewards/std_reward": std_reward,
            "rewards/min_reward": min_reward,
            "rewards/max_reward": max_reward,
            "rewards/mean_episode_length": mean_length,
            "rewards/std_episode_length": std_length,
            "rewards/episodes_recorded": len(self.episode_rewards),
        }, step=self.num_timesteps)
        
        if self.verbose > 0:
            print(f"Step {self.num_timesteps}: Mean reward over {len(self.episode_rewards)} episodes: {mean_reward:.3f} ± {std_reward:.3f}")
    
    def _on_training_end(self) -> None:
        """Called when training ends - log final statistics."""
        if len(self.episode_rewards) > 0:
            self._log_statistics()
            
            # Log final summary
            final_mean = np.mean(self.episode_rewards)
            print(f"Training completed. Final mean reward over {len(self.episode_rewards)} episodes: {final_mean:.3f}")
    
    def _log_statistics(self):
        """Log aggregate reward statistics to wandb."""
        if len(self.episode_rewards) == 0:
            return
            
        rewards_array = np.array(self.episode_rewards)
        lengths_array = np.array(self.episode_lengths)
        
        # Calculate statistics
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)
        min_reward = np.min(rewards_array)
        max_reward = np.max(rewards_array)
        
        mean_length = np.mean(lengths_array)
        std_length = np.std(lengths_array)
        
        # Log to wandb
        wandb.log({
            "rewards/mean_reward": mean_reward,
            "rewards/std_reward": std_reward,
            "rewards/min_reward": min_reward,
            "rewards/max_reward": max_reward,
            "rewards/mean_episode_length": mean_length,
            "rewards/std_episode_length": std_length,
            "rewards/episodes_recorded": len(self.episode_rewards),
        }, step=self.num_timesteps)
        
        if self.verbose > 0:
            print(f"Step {self.num_timesteps}: Mean reward over {len(self.episode_rewards)} episodes: {mean_reward:.3f} ± {std_reward:.3f}")
    
    def _on_training_end(self) -> None:
        """Called when training ends - log final statistics."""
        if len(self.episode_rewards) > 0:
            self._log_statistics()
            
            # Log final summary
            final_mean = np.mean(self.episode_rewards)
            print(f"Training completed. Final mean reward over {len(self.episode_rewards)} episodes: {final_mean:.3f}")
