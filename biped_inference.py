import torch
import numpy as np
import sys
import time
import argparse

# Only import keyboard modules if running with keyboard control
try:
    import termios
    import tty
    import select
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: Keyboard control not available on this system")

from sb3_contrib import RecurrentPPO
from biped_config import get_cfgs
from biped_env import BipedVecEnv
import genesis as gs

# Terminal keyboard input helper
class KBHit:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)

        self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fd, termios.TCSANOW, self.new_term)

    def kbhit(self):
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        return dr != []

    def getch(self):
        return sys.stdin.read(1)

    def set_normal_term(self):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.old_term)


def main():
    parser = argparse.ArgumentParser(description='Biped Robot Inference')
    parser.add_argument('--no-viewer', action='store_true', help='Run without graphical viewer')
    parser.add_argument('--model-path', type=str, default="./logs/biped-walking-sb3-recurrentppo/model_final.zip", 
                       help='Path to the trained model')
    parser.add_argument('--forward-vel', type=float, default=0.5, help='Forward velocity command')
    parser.add_argument('--lateral-vel', type=float, default=0.0, help='Lateral velocity command')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to run')
    parser.add_argument('--keyboard', action='store_true', help='Enable keyboard control')
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(logging_level="warning")

    # Get environment configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg, adaptive_lr_cfg = get_cfgs()

    # Create environment with optional viewer
    show_viewer = not args.no_viewer
    print(f"Creating environment with viewer: {show_viewer}")
    
    env = BipedVecEnv(
        num_envs=1, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg, 
        show_viewer=show_viewer
    )

    # Load the trained model
    model_path = args.model_path
    try:
        model = RecurrentPPO.load(model_path, env=env)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Available model files:")
        import os
        for root, dirs, files in os.walk("./logs"):
            for file in files:
                if file.endswith(".zip"):
                    print(f"  - {os.path.join(root, file)}")
        return

    # Reset environment and get initial observation
    obs = env.reset()
    obs = obs.astype(np.float32)

    # LSTM states for RecurrentPPO
    lstm_states = None
    episode_starts = np.array([True], dtype=bool)

    # Check if keyboard control is requested and available
    use_keyboard = args.keyboard and KEYBOARD_AVAILABLE
    
    if use_keyboard:
        # Initialize keyboard input
        kb = KBHit()
        print("\n" + "="*50)
        print("BIPED ROBOT INFERENCE CONTROL")
        print("="*50)
        print("Controls:")
        print("  W/w - Increase forward velocity")
        print("  S/s - Decrease forward velocity")
        print("  A/a - Increase left velocity")
        print("  D/d - Increase right velocity")
        print("  R/r - Reset velocities to zero")
        print("  Q/q - Quit inference")
        print("="*50)

        # Velocity command variables
        forward_velocity = 0.0
        lateral_velocity = 0.0
        angular_velocity = 0.0
        
        velocity_step = 0.1
        min_vel, max_vel = -1.0, 1.5  # Reasonable velocity limits
        
        print(f"Initial velocities: forward={forward_velocity:.2f}, lateral={lateral_velocity:.2f}")
    else:
        # Use command line velocities
        forward_velocity = args.forward_vel
        lateral_velocity = args.lateral_vel
        angular_velocity = 0.0
        print(f"Running with fixed velocities: forward={forward_velocity:.2f}, lateral={lateral_velocity:.2f}")

    try:
        step_count = 0
        max_steps = args.steps if not use_keyboard else 100000  # Unlimited steps for keyboard mode
        
        while step_count < max_steps:
            # Check for keyboard input if enabled
            if use_keyboard and kb.kbhit():
                c = kb.getch()
                if c.lower() == 'w':
                    forward_velocity = min(forward_velocity + velocity_step, max_vel)
                    print(f"Forward velocity: {forward_velocity:.2f}")
                elif c.lower() == 's':
                    forward_velocity = max(forward_velocity - velocity_step, min_vel)
                    print(f"Forward velocity: {forward_velocity:.2f}")
                elif c.lower() == 'a':
                    lateral_velocity = min(lateral_velocity + velocity_step, max_vel)
                    print(f"Lateral velocity: {lateral_velocity:.2f}")
                elif c.lower() == 'd':
                    lateral_velocity = max(lateral_velocity - velocity_step, min_vel)
                    print(f"Lateral velocity: {lateral_velocity:.2f}")
                elif c.lower() == 'r':
                    forward_velocity = 0.0
                    lateral_velocity = 0.0
                    angular_velocity = 0.0
                    print("Reset all velocities to zero")
                elif c.lower() == 'q':
                    print("Exiting inference.")
                    break

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
                print(f"Episode ended at step {step_count}. Resetting...")
                obs = env.reset()
                lstm_states = None
                episode_starts = np.array([True], dtype=bool)
                obs = obs.astype(np.float32)
                step_count = 0

            step_count += 1

            # Print progress periodically
            if step_count % 100 == 0:
                print(f"Step {step_count}: reward={reward[0]:.3f}")

            # Control simulation speed (50 Hz to match training)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInference stopped by user (Ctrl+C).")

    finally:
        # Clean up
        if use_keyboard:
            kb.set_normal_term()
        env.close()
        print("Environment closed successfully.")


if __name__ == '__main__':
    main()
