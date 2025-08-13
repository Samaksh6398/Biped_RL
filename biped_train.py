# biped_train.py (Corrected)

import argparse
import os
import pickle
import shutil
import signal
import sys
import time
from importlib import metadata

import wandb
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

import genesis as gs

from biped_env import BipedEnv
from biped_config import get_cfgs, get_train_cfg
from ppo_lstm_policy import ActorCriticLSTM, PPOLSTMAgent


class LSTMPPORunner:
    """
    Custom PPO runner that handles LSTM states across steps and episodes.
    """
    
    def __init__(self, env, train_cfg, log_dir, device='cuda'):
        self.env = env
        self.train_cfg = train_cfg
        self.log_dir = log_dir
        self.device = device
        
        # Create policy
        self.policy = ActorCriticLSTM(
            num_obs=env.num_obs,
            num_actions=env.num_actions,
            **train_cfg["policy"]
        ).to(device)
        
        # Create PPO agent
        self.agent = PPOLSTMAgent(
            policy=self.policy,
            device=device,
            num_envs=env.num_envs,
            **train_cfg["algorithm"]
        )
        
        # Initialize LSTM states
        self.lstm_states = self.policy.init_hidden(env.num_envs, device)
        self.masks = torch.ones(env.num_envs, 1, device=device)
        
        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.tot_timesteps = 0
        self.tot_time = 0
        
        # Statistics
        self.ep_infos = []
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        """Main training loop"""
        
        # Initialize environment
        obs, infos = self.env.reset()
        if init_at_random_ep_len:
            self.env.episode_length_buf[:] = torch.randint(
                0, self.env.max_episode_length, (self.env.num_envs,), device=self.device
            )
        
        current_iter = 0
        for it in range(current_iter, num_learning_iterations):
            start_time = time.time()
            
            # Collect rollout
            self.collect_rollout(obs)
            
            # Update policy
            self.agent.update()
            
            # Get new observations after policy update
            obs, infos = self.env.get_observations()
            
            # Logging
            self.log_training_info(it, time.time() - start_time)
            
            # Save model periodically
            if it % self.train_cfg["save_interval"] == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                
        # Save final model
        self.save(os.path.join(self.log_dir, f"model_{num_learning_iterations}.pt"))
    
    def collect_rollout(self, obs):
        """Collect rollout data with LSTM states"""
        
        # Reset storage step counter
        self.agent.storage.step = 0
        
        # Store initial observation and LSTM states
        self.agent.storage.observations[0].copy_(obs)
        self.agent.storage.lstm_hidden_states[0].copy_(self.lstm_states[0])
        self.agent.storage.lstm_cell_states[0].copy_(self.lstm_states[1])
        self.agent.storage.masks[0].copy_(self.masks)
        
        for step in range(self.agent.num_steps_per_env):
            # Forward pass through policy
            with torch.no_grad():
                actions, log_probs, values, new_lstm_states, _ = self.policy(
                    obs, self.lstm_states, self.masks
                )
            
            # Environment step
            obs, rewards, dones, infos, new_masks = self.env.step_lstm(actions, self.lstm_states, self.masks)
            
            # Store experience
            self.agent.storage.insert(
                obs=obs,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                lstm_states=self.lstm_states,
                masks=self.masks
            )
            
            # Update LSTM states and masks for next step
            self.lstm_states = new_lstm_states
            self.masks = new_masks
            
            # Reset LSTM states for completed episodes
            reset_indices = dones.nonzero(as_tuple=False).reshape(-1)
            if len(reset_indices) > 0:
                # Reset LSTM hidden and cell states for completed episodes
                self.lstm_states[0][:, reset_indices, :] = 0.0
                self.lstm_states[1][:, reset_indices, :] = 0.0
            
            # --- START OF FIX ---
            # Collect episode statistics
            # The original code was incorrectly breaking up the dictionary.
            # The correct way is to append the entire 'episode' dictionary.
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])
            # --- END OF FIX ---
            
            self.tot_timesteps += self.env.num_envs
    
    def log_training_info(self, iteration, step_time):
        """Log training statistics"""
        
        if len(self.ep_infos) > 0:
            # This logic is now correct because self.ep_infos contains full dictionaries
            for key in self.ep_infos[0].keys():
                values = [ep_info[key] for ep_info in self.ep_infos if key in ep_info]
                if values:
                    mean_value = sum(values) / len(values)
                    self.writer.add_scalar(f'Episode/{key}', mean_value, iteration)
        
        # Clear episode infos for the next rollout
        self.ep_infos.clear()
        
        # Log training metrics
        self.writer.add_scalar('Training/iteration_time', step_time, iteration)
        self.writer.add_scalar('Training/total_timesteps', self.tot_timesteps, iteration)
        
        # Log to console
        if iteration % 10 == 0:
            print(f"Iteration {iteration: <5} | Total Timesteps: {self.tot_timesteps: <8} | Step Time: {step_time:.3f}s")
    
    def save(self, path):
        """Save model and optimizer state"""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'tot_timesteps': self.tot_timesteps,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model and optimizer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tot_timesteps = checkpoint.get('tot_timesteps', 0)
        print(f"Model loaded from {path}. Resuming at timestep {self.tot_timesteps}.")
    
    def get_inference_policy(self, device=None):
        """Get policy for inference"""
        if device is not None:
            self.policy = self.policy.to(device)
        
        # Import here to avoid circular import
        from biped_eval import InferencePolicyLSTM
        return InferencePolicyLSTM(self.policy, device or self.device)

# ... (main function is unchanged and correct) ...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking-lstm")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=10000)

    # --- W&B Arguments ---
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="biped-rl", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="A name for this specific W&B run.")

    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    # The PPOLSTMAgent expects 'mini_batch_size' and 'lr' arguments, but the
    # config from get_train_cfg() provides 'num_mini_batches' and 'learning_rate'.
    # We perform the conversion here to make them compatible.

    # 1. Update general algorithm parameters for LSTM training
    train_cfg["algorithm"]["num_steps_per_env"] = 32
    train_cfg["algorithm"]["learning_rate"] = 3e-4 # Set desired learning rate

    # 2. Convert 'learning_rate' to 'lr'
    if 'learning_rate' in train_cfg['algorithm']:
        train_cfg['algorithm']['lr'] = train_cfg['algorithm']['learning_rate']
        del train_cfg['algorithm']['learning_rate']

    # 3. Convert 'num_mini_batches' to 'mini_batch_size'
    if 'num_mini_batches' in train_cfg['algorithm']:
        batch_size = args.num_envs * train_cfg['algorithm']['num_steps_per_env']
        # Ensure mini_batch_size is at least 1
        mini_batch_size = max(batch_size // train_cfg['algorithm']['num_mini_batches'], 1)
        train_cfg['algorithm']['mini_batch_size'] = mini_batch_size
        del train_cfg['algorithm']['num_mini_batches']
    else:
        # Fallback if num_mini_batches is not in the config for some reason
        train_cfg['algorithm']['mini_batch_size'] = 256
    
    # 4. Add LSTM-specific policy configuration
    train_cfg["policy"] = {
        "actor_hidden_dims": [512, 256],
        "critic_hidden_dims": [512, 256],
        "lstm_hidden_size": 256,
        "lstm_num_layers": 1,
        "activation": "elu",
        "init_noise_std": 1.0
    }

    # --- Initialize W&B ---
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.exp_name,
            config=train_cfg,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        print(f"W&B logging enabled. Syncing TensorBoard logs from: {log_dir}")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = BipedEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = LSTMPPORunner(env, train_cfg, log_dir, device=gs.device)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Saving current model...')
        runner.save(os.path.join(log_dir, "model_interrupted.pt"))
        if args.wandb:
            wandb.finish()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting training... Press Ctrl+C to stop and save the model.")
    print(f"Logs will be saved to: {log_dir}")
    
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except Exception as e:
        print(f'\nTraining stopped due to error: {e}')
        raise
    finally:
        # Final cleanup and save
        print('Final model save completed.')
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()