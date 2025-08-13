import torch
import genesis as gs

def gs_rand_float(lower, upper, shape, device):
    """
    Generates random floats in a given range.
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class DomainRandomizationManager:
    """
    Manages all domain randomization logic, including physics and sensor randomization.
    """
    def __init__(self, env):
        self.env = env
        self.device = gs.device
        self.dr_cfg = self.env.env_cfg["domain_rand"]

        # Buffers and state for DR
        self.randomized_kp = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device, dtype=gs.tc_float)
        self.orig_kp = torch.tensor([self.env.env_cfg["kp"]] * self.env.num_actions, device=self.device)
        
        self.noise_buffers = {
            'dof_pos': torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device),
            'dof_vel': torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device),
            'lin_vel': torch.zeros((self.env.num_envs, 3), device=self.device),
            'ang_vel': torch.zeros((self.env.num_envs, 3), device=self.device),
            'base_pos': torch.zeros((self.env.num_envs, 3), device=self.device),
            'base_euler': torch.zeros((self.env.num_envs, 3), device=self.device),
            'foot_contact': torch.zeros((self.env.num_envs, 2), device=self.device),
        }
        
        self.randomization_intervals = {
            'motor_strength': 50, 'friction': 100, 'mass': 200,
            'observation_noise': 1, 'foot_contacts': 1, 'motor_backlash': 20,
        }
        self.randomization_step_counter = 0

        # Motor Backlash Buffers
        self.motor_backlash = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device, dtype=gs.tc_float)
        self.motor_backlash_direction = torch.ones((self.env.num_envs, self.env.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_motor_positions = torch.zeros_like(self.env.actions)

        # Foot Contact DR Buffers
        self.contact_thresholds = torch.zeros((self.env.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.contact_noise_scale = torch.zeros((self.env.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.contact_false_positive_prob = torch.zeros((self.env.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.contact_false_negative_prob = torch.zeros((self.env.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.contact_delay_steps = torch.zeros((self.env.num_envs, 2), device=self.device, dtype=torch.long)
        self.contact_delay_buffer = torch.zeros((self.env.num_envs, 2, 5), device=self.device, dtype=gs.tc_float)
        self.contact_delay_idx = torch.zeros((self.env.num_envs,), device=self.device, dtype=torch.long)

    def apply_on_step(self, actions):
        """
        Applies randomizations that occur at each simulation step.
        """
        self.randomization_step_counter += 1
        exec_actions = actions
        if self.dr_cfg["add_motor_backlash"] and self._should_update_randomization('motor_backlash'):
            exec_actions = self._apply_motor_backlash(exec_actions)
        
        if self.dr_cfg["randomize_foot_contacts"]:
            self.env.foot_contacts = self._apply_foot_contact_randomization_optimized(self.env.foot_contacts_raw)
        else:
            self.env.foot_contacts = self.env.foot_contacts_raw.clone()
        
        return exec_actions

    def apply_on_reset(self, envs_idx):
        """
        Applies randomizations that occur when an environment is reset.
        """
        if len(envs_idx) == 0:
            return

        if self.dr_cfg["randomize_motor_strength"] and self._should_update_randomization('motor_strength'):
            strength_scale = gs_rand_float(*self.dr_cfg["motor_strength_range"], (len(envs_idx), self.env.num_actions), self.device)
            self.randomized_kp[envs_idx] = self.orig_kp * strength_scale
            for i, env_idx in enumerate(envs_idx):
                self.env.robot.set_dofs_kp(self.randomized_kp[env_idx], self.env.motors_dof_idx, envs_idx=[env_idx])

        if self.dr_cfg["add_motor_backlash"]:
            backlash_values = gs_rand_float(*self.dr_cfg["backlash_range"], (len(envs_idx), self.env.num_actions), self.device)
            self.motor_backlash[envs_idx] = backlash_values
            self.motor_backlash_direction[envs_idx] = 1.0
            self.last_motor_positions[envs_idx] = 0.0

        if self.dr_cfg["randomize_foot_contacts"]:
            fc_params = self.dr_cfg["foot_contact_params"]
            self.contact_thresholds[envs_idx] = gs_rand_float(*fc_params["contact_threshold_range"], (len(envs_idx), 2), self.device)
            self.contact_noise_scale[envs_idx] = gs_rand_float(*fc_params["contact_noise_range"], (len(envs_idx), 2), self.device)
            self.contact_false_positive_prob[envs_idx] = fc_params["false_positive_rate"]
            self.contact_false_negative_prob[envs_idx] = fc_params["false_negative_rate"]
            self.contact_delay_steps[envs_idx] = torch.randint(fc_params["contact_delay_range"][0], fc_params["contact_delay_range"][1] + 1, (len(envs_idx), 2), device=self.device)
            self.contact_delay_buffer[envs_idx] = 0.0
            self.contact_delay_idx[envs_idx] = 0

    # --- Helper Functions ---
    # (All DR helper functions from the original biped_env.py are moved here)

    def _should_update_randomization(self, randomization_type):
        interval = self.randomization_intervals.get(randomization_type, 1)
        return self.randomization_step_counter % interval == 0

    def _generate_noise_batch(self):
        noise_scales = self.dr_cfg["noise_scales"]
        for noise_type, scale in noise_scales.items():
            if noise_type in self.noise_buffers:
                torch.randn(self.noise_buffers[noise_type].shape, out=self.noise_buffers[noise_type], device=self.device)
                self.noise_buffers[noise_type] *= scale

    def _apply_motor_backlash(self, actions):
        position_diff = actions - self.last_motor_positions
        direction_change = (position_diff * self.motor_backlash_direction) < 0
        backlash_offset = self.motor_backlash * self.motor_backlash_direction
        actions_with_backlash = actions.clone()
        actions_with_backlash[direction_change] += backlash_offset[direction_change]
        self.motor_backlash_direction = torch.sign(position_diff)
        self.motor_backlash_direction[torch.abs(position_diff) < 1e-6] = self.motor_backlash_direction[torch.abs(position_diff) < 1e-6]
        self.last_motor_positions = actions.clone()
        return actions_with_backlash

    def _apply_foot_contact_randomization_optimized(self, raw_contacts):
        contact_detected = raw_contacts > self.contact_thresholds
        if self.dr_cfg["add_observation_noise"]:
            contact_noise = self.noise_buffers['foot_contact']
            randomized_contacts = raw_contacts + contact_noise
        else:
            randomized_contacts = raw_contacts.clone()
        
        false_pos_rand = torch.rand_like(raw_contacts)
        false_neg_rand = torch.rand_like(raw_contacts)
        
        false_pos_mask = (false_pos_rand < self.contact_false_positive_prob) & ~contact_detected
        false_neg_mask = (false_neg_rand < self.contact_false_negative_prob) & contact_detected
        
        randomized_contacts[false_pos_mask] = 1.0
        randomized_contacts[false_neg_mask] = 0.0
        if hasattr(self, 'contact_delay_steps'):
            self._apply_contact_delays_vectorized(randomized_contacts)
        
        return torch.clamp(randomized_contacts, 0.0, 1.0)

    def _apply_contact_delays_vectorized(self, contacts):
        if self.contact_delay_buffer.numel() == 0:
            return contacts
        current_idx = self.contact_delay_idx[0] % self.contact_delay_buffer.shape[2]
        self.contact_delay_buffer[:, :, current_idx] = contacts
        uniform_delays = torch.mode(self.contact_delay_steps.flatten())[0]
        if uniform_delays > 0 and uniform_delays < self.contact_delay_buffer.shape[2]:
            delay_idx = (current_idx - uniform_delays) % self.contact_delay_buffer.shape[2]
            delay_mask = (self.contact_delay_steps == uniform_delays)
            contacts[delay_mask] = self.contact_delay_buffer[:, :, delay_idx][delay_mask]
        non_uniform_mask = (self.contact_delay_steps != uniform_delays)
        if non_uniform_mask.any():
            for env_idx in range(self.env.num_envs):
                for foot_idx in range(2):
                    if non_uniform_mask[env_idx, foot_idx]:
                        delay = self.contact_delay_steps[env_idx, foot_idx]
                        if delay > 0 and delay < self.contact_delay_buffer.shape[2]:
                            delay_idx = (current_idx - delay) % self.contact_delay_buffer.shape[2]
                            contacts[env_idx, foot_idx] = self.contact_delay_buffer[env_idx, foot_idx, delay_idx]
        self.contact_delay_idx += 1
        return contacts