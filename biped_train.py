
import argparse
import os
import pickle
import shutil
import signal
import sys
import torch as th # Use 'th' alias for clarity when mixing with 'torch' from genesis

import wandb
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

import genesis as gs # Ensure genesis is initialized
from biped_env import BipedVecEnv # Import the new VecEnv wrapper
from biped_config import get_cfgs # Get environment-specific configurations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking-sb3-recurrentppo")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=int, default=100_000_000, help="Total timesteps for training.")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="biped-rl-sb3", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="A name for this specific W&B run.")

    args = parser.parse_args()

    # Initialize Genesis simulator first (critical for device setup)
    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    
    # Clean previous logs for a fresh run
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Get environment configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Save environment configs for later inspection/evaluation
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))

    # Create the vectorized environment using the SB3-compatible wrapper
    env = BipedVecEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        show_viewer=False, # Set to True for one environment to visualize during training
        device=gs.device # Pass Genesis device to the environment wrapper
    )

    # --- Stable Baselines3 RecurrentPPO Configuration ---
    # These parameters are chosen to be roughly equivalent to the original rsl-rl config
    n_steps = 32 # Number of steps each environment takes before a policy update
    # Batch size and mini-batch size calculations from original config logic:
    # `num_steps_per_env` in rsl-rl is `n_steps` in SB3.
    # `num_mini_batches` in rsl-rl is implicitly handled by `batch_size` and `n_epochs` in SB3.
    # If original had 4 mini-batches, then SB3's batch_size should be (num_envs * n_steps) // 4.
    batch_size = (args.num_envs * n_steps) // 4 # Total number of samples per policy update step
    
    # Policy kwargs specific for MlpLstmPolicy to mimic original network structure
    policy_kwargs = dict(
        activation_fn=th.nn.ELU, # Use ELU activation
        # net_arch defines the MLP layers before and after the LSTM
        # Example: [dict(pi=[512, 256], vf=[512, 256])] for separate actor/critic MLPs
        # Here, `net_arch` specifies layers *before* LSTM for both actor and critic
        # which is somewhat different from the rsl-rl structure.
        # SB3 `MlpLstmPolicy`'s `net_arch` is applied to observations BEFORE LSTM, and then
        # an additional MLP is applied AFTER LSTM.
        # To mimic [512, 256] actor/critic hidden dims, and [256, 128] encoder,
        # we set `net_arch` for the encoder part.
        # The `lstm_hidden_size` and `n_lstm_layers` are direct mappings.
        net_arch=[dict(pi=[256, 128], vf=[256, 128])], # For the observation encoder
        lstm_hidden_size=256,
        n_lstm_layers=1,
        # Default initialization for SB3 is usually good.
        # Initial log_std is usually set via `log_std_init` in PPO if needed, not directly in policy_kwargs.
        # It's not a direct 1:1 mapping for `init_noise_std`.
    )

    # Callbacks for model saving and optional W&B logging
    # Save a model every 500,000 timesteps (or more frequently if num_envs * n_steps is large)
    save_freq_steps = max(500_000 // args.num_envs, n_steps) 
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_steps, # Frequency in timesteps
        save_path=log_dir,
        name_prefix="biped_model",
        save_replay_buffer=False, # No replay buffer for on-policy PPO
        save_vecnormalize=False, # Not using VecNormalize currently
    )
    callbacks = [checkpoint_callback]

    if args.wandb:
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.exp_name,
            sync_tensorboard=True, # Sync SB3's TensorBoard logs to W&B
            monitor_gym=True,      # Monitor Gymnasium environments
            save_code=True,        # Save code to W&B
            config={               # Log SB3 hyperparameters to W&B config
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": 5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 1.0,
                "learning_rate": 3e-4,
                "max_grad_norm": 1.0,
                "policy_type": "MlpLstmPolicy",
                "policy_kwargs_net_arch_pi": policy_kwargs['net_arch'][0]['pi'],
                "policy_kwargs_net_arch_vf": policy_kwargs['net_arch'][0]['vf'],
                "policy_kwargs_lstm_hidden_size": policy_kwargs['lstm_hidden_size'],
                "policy_kwargs_n_lstm_layers": policy_kwargs['n_lstm_layers'],
                **env_cfg # Log environment config as well
            }
        )
        wandb_callback = WandbCallback(
            model_save_path=f"models/{run.id}", # Save models to W&B's run directory
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    # Create the RecurrentPPO model
    model = RecurrentPPO(
        "MlpLstmPolicy", # Policy type
        env,             # Environment
        verbose=1,       # Log progress to stdout
        n_steps=n_steps, # Number of steps to run for each environment per update
        batch_size=batch_size, # Number of samples in a batch for training
        n_epochs=5,      # Number of PPO training epochs
        gamma=0.99,      # Discount factor
        gae_lambda=0.95, # Factor for Generalized Advantage Estimator
        clip_range=0.2,  # PPO clipping parameter
        ent_coef=0.01,   # Entropy coefficient for exploration
        vf_coef=1.0,     # Value function loss coefficient
        learning_rate=3e-4, # Learning rate
        max_grad_norm=1.0,  # Gradient clipping
        policy_kwargs=policy_kwargs, # Policy network specific arguments
        tensorboard_log=log_dir, # Log to TensorBoard (also synced to W&B if enabled)
        device=gs.device # Use the same device as Genesis
    )

    # Signal handler for graceful interruption (Ctrl+C)
    def signal_handler(sig, frame):
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Saving current model...')
        model.save(os.path.join(log_dir, "model_interrupted.zip"))
        if args.wandb:
            wandb.finish()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Press Ctrl+C to stop and save the model.")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True, # Show a progress bar during training
        )
    except Exception as e:
        print(f'\nTraining stopped due to error: {e}')
        raise
    finally:
        # Save final model regardless of interruption or completion
        model.save(os.path.join(log_dir, "model_final.zip"))
        print('Training finished. Final model saved.')
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
