# biped_rewards.py (Corrected)

import torch
import numpy as np
import genesis as gs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biped_env import BipedEnv


class RewardHandler:
    def __init__(self, env: 'BipedEnv'):
        self.env = env
        self.reward_scales = self.env.reward_cfg["reward_scales"]
        self.reward_cfg = self.env.reward_cfg
        self.env_cfg = self.env.env_cfg
        self.device = self.env.device
        
        # --- FIX: Initialize the attribute ---
        self.reward_components = {}

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions = dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.env.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)

    def compute_rewards(self):
        # --- START OF FIX ---
        # This is the corrected logic. We now populate self.reward_components
        # which makes the RewardHandler self-contained and fixes the AttributeError.
        
        self.env.rew_buf[:] = 0.0
        self.reward_components.clear() # Clear at the start of each step

        for name, reward_func in self.reward_functions.items():
            # Check if the reward is enabled in the config
            if self.reward_cfg.get("reward_enables", {}).get(name, True):
                # 1. Calculate the raw reward value
                raw_rew = reward_func()
                
                # 2. Store the raw, unscaled reward for potential debugging
                self.reward_components[name] = raw_rew
                
                # 3. Scale the reward for use
                scaled_rew = raw_rew * self.reward_scales[name]
                
                # 4. Add to the total reward buffer for the agent
                self.env.rew_buf += scaled_rew
                
                # 5. Add the scaled reward to the episode sums for logging
                if name in self.env.episode_sums:
                    self.env.episode_sums[name] += scaled_rew
        # --- END OF FIX ---

    def _reward_lin_vel_z(self):
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        return torch.zeros((self.env.num_envs,), device=self.device, dtype=gs.tc_float)

    def _reward_forward_velocity(self):
        v_target = self.reward_cfg.get("forward_velocity_target", 0.5)
        vel_error = torch.square(self.env.base_lin_vel[:, 0] - v_target)
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-vel_error / sigma)

    def _reward_tracking_lin_vel_x(self):
        lin_vel_error = torch.square(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        lin_vel_error = torch.square(self.env.commands[:, 1] - self.env.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_alive_bonus(self):
        return torch.ones((self.env.num_envs,), device=self.device, dtype=gs.tc_float)

    def _reward_fall_penalty(self):
        fall_condition = (
            (torch.abs(self.env.base_euler[:, 0]) > self.env_cfg.get("fall_roll_threshold", 30.0)) |  # Roll > 30 degrees
            (torch.abs(self.env.base_euler[:, 1]) > self.env_cfg.get("fall_pitch_threshold", 30.0))   # Pitch > 30 degrees
        )
        return torch.where(
            fall_condition,
            torch.ones((self.env.num_envs,), device=self.device, dtype=gs.tc_float),  # Apply penalty
            torch.zeros((self.env.num_envs,), device=self.device, dtype=gs.tc_float)  # No penalty
        )

    def _reward_torso_stability(self):
        orientation_error = torch.sum(torch.square(self.env.base_euler[:, :2]), dim=1)  # φ² + θ² (roll² + pitch²)
        k_stability = self.reward_cfg.get("stability_factor", 1.0)
        return torch.exp(-k_stability * orientation_error)

    def _reward_height_maintenance(self):
        z_target = self.reward_cfg.get("height_target", 0.35)
        height_error = torch.square(z_target - self.env.base_pos[:, 2])
        return -height_error  # Return negative error (will be scaled by negative weight in config)

    def _reward_joint_movement(self):
        joint_vel_magnitude = torch.sum(torch.abs(self.env.dof_vel), dim=1)
        movement_threshold = self.reward_cfg.get("movement_threshold", 0.1)
        movement_scale = self.reward_cfg.get("movement_scale", 1.0)

        return torch.clamp(joint_vel_magnitude * movement_scale, 0.0, movement_threshold)