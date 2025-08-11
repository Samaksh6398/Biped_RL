import argparse
import os
import pickle
from importlib import metadata
import threading
import time

import torch

# This script uses the rsl-rl-lib for loading the PPO runner and policy.
# It checks for the correct version to ensure compatibility.
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

# Genesis is the physics simulator used for the environment.
import genesis as gs

# This imports the custom environment definition for the bipedal robot.
from biped_env import BipedEnv


class KeyboardController:
    """
    Handles keyboard input for controlling forward velocity commands.
    """
    def __init__(self):
        self.forward_vel = 0.0  # Current forward velocity command
        self.velocity_step = 0.1  # Velocity increment/decrement step
        self.max_velocity = 1.0  # Maximum forward velocity
        self.min_velocity = -0.5  # Maximum backward velocity
        self.running = True
        
    def print_controls(self):
        """Print the control instructions."""
        print("\n" + "="*60)
        print("🎮 KEYBOARD CONTROLS:")
        print("="*60)
        print("W / ↑  : Increase forward velocity (+{:.1f} m/s)".format(self.velocity_step))
        print("S / ↓  : Decrease forward velocity (-{:.1f} m/s)".format(self.velocity_step))
        print("SPACE  : Stop (set velocity to 0.0 m/s)")
        print("Q      : Quit inference")
        print("="*60)
        print(f"Current velocity: {self.forward_vel:.2f} m/s")
        print(f"Range: [{self.min_velocity:.1f}, {self.max_velocity:.1f}] m/s")
        print("="*60)
        
    def keyboard_listener(self):
        """
        Listen for keyboard input in a separate thread.
        Note: This is a simple implementation. For more robust keyboard handling,
        consider using libraries like 'keyboard' or 'pynput'.
        """
        try:
            while self.running:
                try:
                    # Simple input-based control (blocking)
                    command = input("Enter command (w/s/space/q) or press Enter to continue: ").lower().strip()
                    
                    if command in ['w', 'up']:
                        self.forward_vel = min(self.forward_vel + self.velocity_step, self.max_velocity)
                        print(f"🚀 Forward velocity: {self.forward_vel:.2f} m/s")
                    elif command in ['s', 'down']:
                        self.forward_vel = max(self.forward_vel - self.velocity_step, self.min_velocity)
                        print(f"🐌 Forward velocity: {self.forward_vel:.2f} m/s")
                    elif command in ['space', ' ', '']:
                        self.forward_vel = 0.0
                        print(f"⏸️  Stopped: {self.forward_vel:.2f} m/s")
                    elif command in ['q', 'quit', 'exit']:
                        print("🛑 Quitting inference...")
                        self.running = False
                        break
                    
                except (EOFError, KeyboardInterrupt):
                    print("\n🛑 Keyboard interrupt received. Quitting...")
                    self.running = False
                    break
                    
        except Exception as e:
            print(f"⚠️  Keyboard listener error: {e}")
            self.running = False
    
    def get_velocity_command(self):
        """Get the current velocity command."""
        return self.forward_vel
    
    def is_running(self):
        """Check if the controller is still running."""
        return self.running


def main():
    """
    Main function to load a trained policy and run inference with keyboard control.
    """
    # --- 1. Argument Parsing ---
    # Sets up command-line arguments to specify which trained model to load.
    parser = argparse.ArgumentParser(description="Run inference for the bipedal robot with keyboard control.")
    parser.add_argument(
        "-e", 
        "--exp_name", 
        type=str, 
        default="biped-walking",
        help="The name of the experiment, used to find the log directory."
    )
    parser.add_argument(
        "--ckpt", 
        type=int, 
        default=100,
        help="The checkpoint number of the model to load (e.g., 100 for 'model_100.pt')."
    )
    args = parser.parse_args()

    # --- 2. Initialization ---
    # Initialize the Genesis simulator.
    gs.init()

    # --- 3. Load Configurations and Model ---
    # Construct the path to the directory where logs and models are saved.
    log_dir = f"logs/{args.exp_name}"
    
    # Check if the specified log directory and model file exist.
    config_path = f"{log_dir}/cfgs.pkl"
    model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    print(f"Loading configurations from: {config_path}")
    print(f"Loading model checkpoint from: {model_path}")

    # Load the configuration files saved during training.
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(config_path, "rb"))
    
    # For inference, we don't need to calculate rewards. Clearing the reward scales
    # can prevent unnecessary computations.
    reward_cfg["reward_scales"] = {}
    
    # Disable domain randomization for inference (cleaner simulation)
    if "domain_rand" in env_cfg:
        env_cfg["domain_rand"]["randomize_motor_strength"] = False
        env_cfg["domain_rand"]["randomize_friction"] = False
        env_cfg["domain_rand"]["randomize_mass"] = False
        env_cfg["domain_rand"]["push_robot"] = False

    # --- 4. Create Environment ---
    # Instantiate the bipedal environment with the loaded configurations.
    # We use num_envs=1 because we are only visualizing one robot.
    # show_viewer=True opens the simulation window.
    env = BipedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # --- 5. Load Policy ---
    # The OnPolicyRunner is used here as a convenient way to load the model
    # and get the policy, even though we are not training.
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(model_path)
    
    # Get the policy in "inference mode". This prepares the neural network
    # for efficient execution without tracking gradients.
    policy = runner.get_inference_policy(device=gs.device)

    # --- 6. Setup Keyboard Controller ---
    keyboard_controller = KeyboardController()
    keyboard_controller.print_controls()
    
    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=keyboard_controller.keyboard_listener, daemon=True)
    keyboard_thread.start()

    # --- 7. Inference Loop with Keyboard Control ---
    # Reset the environment to get the first observation.
    obs, _ = env.reset()
    
    print("\n🚀 Inference started with keyboard control!")
    print("💡 The robot will follow your velocity commands.")
    print("🔄 Commands are updated in real-time.")
    
    step_count = 0
    last_print_time = time.time()
    
    # The context `torch.no_grad()` is a performance optimization that tells PyTorch
    # not to compute gradients, making inference faster.
    with torch.no_grad():
        # Loop until keyboard controller signals to quit
        while keyboard_controller.is_running():
            try:
                # Get current velocity command from keyboard controller
                forward_vel_cmd = keyboard_controller.get_velocity_command()
                
                # Update the environment's velocity commands
                # Commands format: [lin_vel_x, lin_vel_y, ang_vel]
                env.commands[:, 0] = forward_vel_cmd  # Forward velocity
                env.commands[:, 1] = 0.0  # No sideways velocity
                env.commands[:, 2] = 0.0  # No angular velocity
                
                # The policy takes the current observation as input and returns the
                # optimal action (motor commands) as output.
                actions = policy(obs)
                
                # The environment executes the action and returns the next state.
                obs, rews, dones, infos = env.step(actions)
                
                # Print status periodically
                step_count += 1
                current_time = time.time()
                if current_time - last_print_time > 2.0:  # Print every 2 seconds
                    actual_vel = env.base_lin_vel[0, 0].item()  # Actual forward velocity
                    print(f"📊 Step: {step_count:6d} | Command: {forward_vel_cmd:+.2f} m/s | Actual: {actual_vel:+.2f} m/s")
                    last_print_time = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n🛑 Received keyboard interrupt. Stopping inference...")
                break
            except Exception as e:
                print(f"⚠️  Error during inference: {e}")
                break
    
    print("\n✅ Inference completed. Goodbye!")
    keyboard_controller.running = False


if __name__ == "__main__":
    main()

"""
# How to run this script with keyboard control:
# ============================================
# 
# 1. Basic usage:
#    python biped_inference.py -e biped-walking --ckpt 100
#
# 2. With different experiment:
#    python biped_inference.py -e my-experiment --ckpt 200
#
# 3. Keyboard Controls (during inference):
#    W or ↑     : Increase forward velocity (+0.1 m/s)
#    S or ↓     : Decrease forward velocity (-0.1 m/s)  
#    SPACE      : Stop robot (velocity = 0.0 m/s)
#    Q          : Quit inference
#
# 4. Features:
#    - Real-time velocity command control
#    - Visual feedback of commanded vs actual velocity
#    - Smooth command transitions
#    - Domain randomization disabled for clean simulation
#
# 5. Notes:
#    - The robot will follow your velocity commands in real-time
#    - Commands are displayed every 2 seconds
#    - Use the simulation viewer to see the robot's movement
#    - Close the terminal or press 'Q' to exit
#
"""
