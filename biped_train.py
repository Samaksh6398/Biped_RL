import argparse
import os
import pickle
import shutil
import signal
import sys
from importlib import metadata

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
from wandb_logger import WandbLogger  # Import the new W&B logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=1000)
    parser.add_argument("--max_iterations", type=int, default=999999)  # Runs until Ctrl+C
    
    # --- W&B Arguments ---
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="biped-rl", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user or team).")
    
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

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

    # --- Initialize Logger ---
    logger = None
    if args.wandb:
        # If the --wandb flag is used, create an instance of our custom logger
        logger = WandbLogger(
            train_cfg=train_cfg,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )

    # Pass the logger to the OnPolicyRunner
    runner = OnPolicyRunner(env, train_cfg, log_dir, logger=logger, device=gs.device)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Saving current model...')
        if logger:
            logger.finish()  # Ensure W&B run is finished properly
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
        if logger:
            logger.finish()


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
# 4. (Optional) Specify a project and entity:
#    python biped_train.py -e biped-walking -B 2048 --wandb --wandb_project my-biped-project --wandb_entity my-username
#
"""