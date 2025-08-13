import argparse
import os
import pickle
from importlib import metadata

import torch

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
from ppo_lstm_policy import ActorCriticLSTM


class InferencePolicyLSTM:
    """
    Wrapper for LSTM policy inference that manages LSTM states.
    """
    
    def __init__(self, policy, device, num_envs=1):
        self.policy = policy
        self.device = device
        self.num_envs = num_envs
        
        # Initialize LSTM states for inference
        self.lstm_states = policy.init_hidden(num_envs, device)
        self.masks = torch.ones(num_envs, 1, device=device)
    
    def __call__(self, observations, deterministic=True):
        """Run inference with LSTM state management"""
        
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
            
        actions, new_lstm_states = self.policy.act(
            observations, self.lstm_states, self.masks, deterministic=deterministic
        )
        
        # Update LSTM states
        self.lstm_states = new_lstm_states
        
        return actions.squeeze(0) if actions.shape[0] == 1 else actions
    
    def reset_states(self, env_indices=None):
        """Reset LSTM states for specified environments (or all if None)"""
        
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_indices, torch.Tensor):
            env_indices = torch.tensor(env_indices, device=self.device)
        
        # Reset LSTM states for specified environments
        self.lstm_states[0][:, env_indices, :] = 0.0
        self.lstm_states[1][:, env_indices, :] = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = BipedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,  # Enable viewer for evaluation
    )

    # Create LSTM policy
    policy = ActorCriticLSTM(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        **train_cfg["policy"]
    ).to(gs.device)
    
    # Load trained model
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    checkpoint = torch.load(resume_path, map_location=gs.device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {resume_path}")
    
    # Create inference wrapper
    inference_policy = InferencePolicyLSTM(policy, gs.device, num_envs=1)

    # Evaluation loop
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    print(f"Starting evaluation for {args.num_episodes} episodes...")
    
    with torch.no_grad():
        while episode_count < args.num_episodes:
            actions = inference_policy(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = env.step(actions)
            
            episode_reward += rewards.item()
            episode_length += 1
            
            if dones:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                print(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}")
                
                # Reset for next episode
                obs, _ = env.reset()
                inference_policy.reset_states()  # Reset LSTM states
                episode_reward = 0.0
                episode_length = 0
    
    # Print evaluation results
    if episode_rewards:
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        mean_length = sum(episode_lengths) / len(episode_lengths)
        print(f"\nEvaluation Results:")
        print(f"Mean Episode Reward: {mean_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        print(f"Min/Max Reward: {min(episode_rewards):.2f}/{max(episode_rewards):.2f}")
    else:
        print("No episodes completed during evaluation.")


if __name__ == "__main__":
    main()

"""
# evaluation
python biped_eval.py -e biped-walking --ckpt 100 --num_episodes 5 --deterministic
"""