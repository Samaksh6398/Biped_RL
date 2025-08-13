import argparse
import os
import pickle
import shutil
import signal
import sys
from importlib import metadata

# --- W&B Import ---
import wandb

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from biped_env import BipedEnv
from biped_config import get_cfgs, get_train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=1000)
    parser.add_argument("--max_iterations", type=int, default=999999)  # Runs until Ctrl+C

    # --- W&B Arguments ---
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="biped-rl", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="A name for this specific W&B run.")

    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    
    # --- CORRECTED SECTION ---
    # This is the corrected part. We call the two functions separately.
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    # --- END OF CORRECTION ---

    # --- Initialize W&B ---
    if args.wandb:
        # If the --wandb flag is used, initialize a W&B run
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.exp_name, # Use a run name if provided
            config=train_cfg,
            sync_tensorboard=True,  # This is the key change!
            monitor_gym=True,       # Automatically log videos of the environment
            save_code=True,         # Save the main script to W&B
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

    # The runner is initialized without any logger parameter
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Saving current model...')
        if args.wandb:
            wandb.finish()  # Ensure W&B run is finished properly
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting training... Press Ctrl+C to stop and save the model.")
    print(f"Logs will be saved to: {log_dir}")
    
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Final model save completed.')
    except Exception as e:
        print(f'\nTraining stopped due to error: {e}')
        raise
    finally:
        # Final cleanup
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()

"""
# How to run with W&B logging:
# ============================
#
# 1. Install wandb:
#    pip install wandb
#
# 2. Login to your W&B account:
#    wandb login
#
# 3. Run training with the --wandb flag:
#    python biped_train.py -e biped-walking -B 2048 --wandb
#
# 4. (Optional) Specify a project, entity, and run name:
#    python biped_train.py -e biped-walking -B 2048 --wandb --wandb_project my-biped-project --wandb_entity my-username --wandb_run_name "first_test_run"
#
"""