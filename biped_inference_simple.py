import torch
import numpy as np
import time

from sb3_contrib import RecurrentPPO
from biped_config import get_cfgs
from biped_env import BipedVecEnv
import genesis as gs

def main():
    print("Starting biped inference...")
    
    # Initialize Genesis
    gs.init(logging_level="warning")

    # Get environment configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg, adaptive_lr_cfg = get_cfgs()
    print("Loaded configurations")

    # Create environment with viewer enabled for inference
    env = BipedVecEnv(
        num_envs=1, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg, 
        show_viewer=True  # Enable viewer for visual feedback
    )
    print("Created environment")

    # Load the trained model
    model_path = "./logs/biped-walking-sb3-recurrentppo/model_final.zip"
    try:
        model = RecurrentPPO.load(model_path, env=env)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    # Reset environment and get initial observation
    obs = env.reset()
    obs = obs.astype(np.float32)
    print(f"Initial observation shape: {obs.shape}")

    # LSTM states for RecurrentPPO
    lstm_states = None
    episode_starts = np.array([True], dtype=bool)

    print("Starting inference loop...")
    
    # Set desired velocities
    forward_velocity = 0.5  # Walk forward at 0.5 m/s
    lateral_velocity = 0.0
    angular_velocity = 0.0

    try:
        for step in range(1000):  # Run for 1000 steps (20 seconds)
            # Update the environment's command directly
            if hasattr(env.biped_env, 'commands'):
                env.biped_env.commands[0, 0] = forward_velocity  # Linear velocity X
                env.biped_env.commands[0, 1] = lateral_velocity   # Linear velocity Y
                env.biped_env.commands[0, 2] = angular_velocity   # Angular velocity Z

            # Predict action using the trained model
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )

            # Step the environment
            obs, reward, done, info = env.step(action)
            obs = obs.astype(np.float32)

            # Handle episode termination
            episode_starts = np.array([done[0]], dtype=bool)

            if done[0]:
                print(f"Episode ended at step {step}. Resetting...")
                obs = env.reset()
                lstm_states = None
                episode_starts = np.array([True], dtype=bool)
                obs = obs.astype(np.float32)

            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: reward={reward[0]:.3f}")

            # Control simulation speed (50 Hz to match training)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInference stopped by user (Ctrl+C).")

    finally:
        # Clean up
        env.close()
        print("Environment closed successfully.")


if __name__ == '__main__':
    main()
