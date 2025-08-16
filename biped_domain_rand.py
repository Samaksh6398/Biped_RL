import torch
import genesis as gs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biped_env import BipedEnv


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class DomainRandomizationHandler:
    def __init__(self, env: 'BipedEnv'):
        self.env = env
        self.env_cfg = self.env.env_cfg
        self.device = self.env.device

    def apply_on_step(self, exec_actions):
        # Apply motor backlash if enabled
        if self.env_cfg["domain_rand"]["add_motor_backlash"]:
            exec_actions = self._apply_motor_backlash(exec_actions)

        # Apply external perturbations (robot pushing)
        dr_cfg = self.env_cfg["domain_rand"]
        if dr_cfg["push_robot"] and self._should_update_randomization('push_robot'):
            # This part is currently disabled in the config
            pass

        return exec_actions

    def apply_on_reset(self, envs_idx):
        if len(envs_idx) == 0:
            return

        dr_cfg = self.env_cfg["domain_rand"]

        # Vectorized motor strength randomization
        if dr_cfg["randomize_motor_strength"] and self._should_update_randomization('motor_strength'):
            strength_scale = gs_rand_float(
                dr_cfg["motor_strength_range"][0],
                dr_cfg["motor_strength_range"][1],
                (len(envs_idx), self.env.num_actions),
                self.device
            )
            self.env.randomized_kp[envs_idx] = self.env.orig_kp * strength_scale
            for i, env_idx in enumerate(envs_idx):
                self.env.robot.set_dofs_kp(
                    self.env.randomized_kp[env_idx],
                    self.env.motors_dof_idx,
                    envs_idx=[env_idx]
                )

        # Friction randomization
        if dr_cfg["randomize_friction"] and self._should_update_randomization('friction'):
            try:
                friction = gs_rand_float(
                    dr_cfg["friction_range"][0],
                    dr_cfg["friction_range"][1],
                    (len(envs_idx),),
                    self.device
                )
                if hasattr(self.env.robot, 'set_friction_batch'):
                    self.env.robot.set_friction_batch(friction, envs_idx=envs_idx)
                else:
                    for i, env_idx in enumerate(envs_idx):
                        if hasattr(self.env.robot, 'set_friction'):
                            self.env.robot.set_friction(friction[i].item())
            except (AttributeError, TypeError) as e:
                if not hasattr(self.env, '_friction_warning_shown'):
                    print(f"Warning: Friction randomization not supported: {e}")
                    self.env._friction_warning_shown = True

        # Mass randomization
        if dr_cfg["randomize_mass"] and self._should_update_randomization('mass'):
            try:
                torso_link = None
                for link in self.env.robot.links:
                    if link.name in ["revolute_torso", "base_link", "torso_link", "base", "servo1"]:
                        torso_link = link
                        break
                if torso_link is not None:
                    base_mass = 1.0
                    added_mass = gs_rand_float(
                        dr_cfg["added_mass_range"][0],
                        dr_cfg["added_mass_range"][1],
                        (len(envs_idx),),
                        self.device
                    )
                    if hasattr(self.env.robot, 'set_link_mass_batch'):
                        new_masses = base_mass + added_mass
                        self.env.robot.set_link_mass_batch(torso_link.idx, new_masses, envs_idx=envs_idx)
                    else:
                        for i, env_idx in enumerate(envs_idx):
                            new_mass = base_mass + added_mass[i].item()
                            # ... (rest of the individual mass setting logic)
                else:
                    if not hasattr(self.env, '_mass_warning_shown'):
                        print("Warning: Torso link not found for mass randomization")
                        self.env._mass_warning_shown = True
            except (AttributeError, TypeError) as e:
                if not hasattr(self.env, '_mass_api_warning_shown'):
                    print(f"Warning: Mass randomization not supported: {e}")
                    self.env._mass_api_warning_shown = True
        
        # Reset motor backlash buffers
        if self.env_cfg["domain_rand"]["add_motor_backlash"]:
            backlash_values = gs_rand_float(
                self.env_cfg["domain_rand"]["backlash_range"][0],
                self.env_cfg["domain_rand"]["backlash_range"][1],
                (len(envs_idx), self.env.num_actions),
                self.device
            )
            self.env.motor_backlash[envs_idx] = backlash_values
            self.env.motor_backlash_direction[envs_idx] = 1.0
            self.env.last_motor_positions[envs_idx] = 0.0

        # Reset foot contact domain randomization parameters
        if self.env_cfg["domain_rand"]["randomize_foot_contacts"]:
            fc_params = self.env_cfg["domain_rand"]["foot_contact_params"]
            self.env.contact_thresholds[envs_idx] = gs_rand_float(*fc_params["contact_threshold_range"], (len(envs_idx), 2), self.device)
            self.env.contact_noise_scale[envs_idx] = gs_rand_float(*fc_params["contact_noise_range"], (len(envs_idx), 2), self.device)
            self.env.contact_false_positive_prob[envs_idx] = fc_params["false_positive_rate"]
            self.env.contact_false_negative_prob[envs_idx] = fc_params["false_negative_rate"]
            self.env.contact_delay_steps[envs_idx] = torch.randint(*fc_params["contact_delay_range"], (len(envs_idx), 2), device=self.device)
            self.env.contact_delay_buffer[envs_idx] = 0.0
            self.env.contact_delay_idx[envs_idx] = 0

    def _should_update_randomization(self, randomization_type):
        interval = self.env.randomization_intervals.get(randomization_type, 1)
        return self.env.randomization_step_counter % interval == 0

    def _generate_noise_batch(self):
        noise_scales = self.env_cfg["domain_rand"]["noise_scales"]
        for noise_type, scale in noise_scales.items():
            if noise_type in self.env.noise_buffers:
                torch.randn(self.env.noise_buffers[noise_type].shape, out=self.env.noise_buffers[noise_type], device=self.device)
                self.env.noise_buffers[noise_type] *= scale

    def _apply_foot_contact_randomization_optimized(self, raw_contacts):
        contact_detected = raw_contacts > self.env.contact_thresholds
        if self.env_cfg["domain_rand"]["add_observation_noise"]:
            if 'foot_contact' in self.env.noise_buffers:
                contact_noise = self.env.noise_buffers['foot_contact']
            else:
                contact_noise = torch.randn_like(raw_contacts) * self.env.contact_noise_scale
            randomized_contacts = raw_contacts + contact_noise
        else:
            randomized_contacts = raw_contacts.clone()

        false_pos_rand = torch.rand_like(raw_contacts)
        false_neg_rand = torch.rand_like(raw_contacts)
        false_pos_mask = (false_pos_rand < self.env.contact_false_positive_prob) & ~contact_detected
        false_neg_mask = (false_neg_rand < self.env.contact_false_negative_prob) & contact_detected
        randomized_contacts[false_pos_mask] = 1.0
        randomized_contacts[false_neg_mask] = 0.0
        if hasattr(self.env, 'contact_delay_steps'):
            self._apply_contact_delays_vectorized(randomized_contacts)
        return torch.clamp(randomized_contacts, 0.0, 1.0)

    def _apply_contact_delays_vectorized(self, contacts):
        if self.env.contact_delay_buffer.numel() == 0:
            return contacts
        current_idx = self.env.contact_delay_idx[0] % self.env.contact_delay_buffer.shape[2]
        self.env.contact_delay_buffer[:, :, current_idx] = contacts
        uniform_delays = torch.mode(self.env.contact_delay_steps.flatten())[0]
        if uniform_delays > 0 and uniform_delays < self.env.contact_delay_buffer.shape[2]:
            delay_idx = (current_idx - uniform_delays) % self.env.contact_delay_buffer.shape[2]
            delay_mask = (self.env.contact_delay_steps == uniform_delays)
            contacts[delay_mask] = self.env.contact_delay_buffer[:, :, delay_idx][delay_mask]
        non_uniform_mask = (self.env.contact_delay_steps != uniform_delays)
        if non_uniform_mask.any():
            for env_idx in range(self.env.num_envs):
                for foot_idx in range(2):
                    if non_uniform_mask[env_idx, foot_idx]:
                        delay = self.env.contact_delay_steps[env_idx, foot_idx]
                        if delay > 0 and delay < self.env.contact_delay_buffer.shape[2]:
                            delay_idx = (current_idx - delay) % self.env.contact_delay_buffer.shape[2]
                            contacts[env_idx, foot_idx] = self.env.contact_delay_buffer[env_idx, foot_idx, delay_idx]
        self.env.contact_delay_idx += 1
        return contacts

    def _apply_motor_backlash(self, actions):
        position_diff = actions - self.env.last_motor_positions
        direction_change = (position_diff * self.env.motor_backlash_direction) < 0
        backlash_offset = self.env.motor_backlash * self.env.motor_backlash_direction
        actions_with_backlash = actions.clone()
        actions_with_backlash[direction_change] += backlash_offset[direction_change]
        self.env.motor_backlash_direction = torch.sign(position_diff)
        self.env.motor_backlash_direction[torch.abs(position_diff) < 1e-6] = self.env.motor_backlash_direction[torch.abs(position_diff) < 1e-6]
        self.env.last_motor_positions = actions.clone()
        return actions_with_backlash
