import os
import torch._dynamo

# Disable TorchInductor/Triton for old GPUs (GTX 1080 Ti)
torch._dynamo.config.suppress_errors = True
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"  # Safe headless rendering
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"

import argparse
import pickle
import shutil
import signal
import sys
import torch as th
import numpy as np
from collections import deque

import wandb
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

import genesis as gs
from biped_env import BipedVecEnv
from biped_config import get_cfgs
from adaptive_lr_callback import ImprovedKLAdaptiveLRCallback
from wandb_reward_callback import WandbRewardCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking-sb3-recurrentppo")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=int, default=200_000_000)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="biped-rl-sb3")                                        
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--adaptive_lr", action="store_true")
    args = parser.parse_args()

    # Initialize Genesis in fully headless mode (CPU fallback for rendering)
    gs.init(logging_level="warning", render_device="cpu")

    log_dir = f"logs/{args.exp_name}"
    if not (args.continue_training or args.load_model):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, adaptive_lr_cfg = get_cfgs()
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, adaptive_lr_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))

    env = BipedVecEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device=gs.device
    )

    n_steps = 32
    batch_size = (args.num_envs * n_steps) // 4

    policy_kwargs = dict(
        activation_fn=th.nn.ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        lstm_hidden_size=512,
        n_lstm_layers=2
    )

    save_freq_steps = max(500_000 // args.num_envs, n_steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path=log_dir,
        name_prefix="biped_model",
    )
    callbacks = [checkpoint_callback]

    if args.adaptive_lr:
        adaptive_lr_callback = ImprovedKLAdaptiveLRCallback(
            target_kl=adaptive_lr_cfg["target_kl"],
            lr_factor=adaptive_lr_cfg["lr_factor"],
            patience=adaptive_lr_cfg["patience"],
            smoothing_window=adaptive_lr_cfg["smoothing_window"],
            min_lr=adaptive_lr_cfg["min_lr"],
            adaptation_threshold=adaptive_lr_cfg["adaptation_threshold"],
            verbose=adaptive_lr_cfg["verbose"]
        )
        callbacks.append(adaptive_lr_callback)

    if args.wandb:
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.exp_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)
        callbacks.append(wandb_callback)
        reward_callback = WandbRewardCallback(log_freq=1000, mean_window=100, verbose=1)
        callbacks.append(reward_callback)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=1.0,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device=gs.device
    )

    if args.load_model:
        if os.path.exists(args.load_model):
            model = RecurrentPPO.load(args.load_model, env=env, device=gs.device)
        else:
            print(f"Error: Model not found: {args.load_model}")
            sys.exit(1)

    if args.continue_training:
        model_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.zip')]
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            model = RecurrentPPO.load(latest_model, env=env, device=gs.device)

    def signal_handler(sig, frame):
        print('\nTraining interrupted by user.')
        model.save(os.path.join(log_dir, "model_interrupted.zip"))
        if args.wandb:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print(f"Starting training for {args.total_timesteps} timesteps... Logs: {log_dir}")

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    except Exception as e:
        print(f'\nTraining stopped due to error: {e}')
        raise
    finally:
        model.save(os.path.join(log_dir, "model_final.zip"))
        if args.wandb:
            wandb.finish()
        print('Training finished. Final model saved.')

if __name__ == "__main__":
    main()
