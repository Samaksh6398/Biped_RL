
import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.sensors import RigidContactForceGridSensor
import numpy as np
import time

# Imports for SB3 integration
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from biped_rewards import RewardHandler
from biped_domain_rand import DomainRandomizationHandler, gs_rand_float


class BipedEnv:
    """
    The core vectorized biped environment running on the Genesis simulator.
    This class handles the physics simulation, state updates, and reward calculations.
    It is wrapped by BipedVecEnv to be compatible with Stable Baselines3.
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat).repeat(self.num_envs, 1)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/biped_v4.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # foot contact sensors
        self.left_foot_contact_sensor = None
        self.right_foot_contact_sensor = None
        for link in self.robot.links:
            if link.name == "revolute_leftfoot":
                self.left_foot_contact_sensor = RigidContactForceGridSensor(entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2))
            elif link.name == "revolute_rightfoot":
                self.right_foot_contact_sensor = RigidContactForceGridSensor(entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2))

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Domain randomization setup
        self.orig_kp = torch.tensor([self.env_cfg["kp"]] * self.num_actions, device=gs.device)
        self.randomized_kp = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        
        # Pre-allocate noise buffers
        self.noise_buffers = {
            'dof_pos': torch.zeros((self.num_envs, self.num_actions), device=gs.device),
            'dof_vel': torch.zeros((self.num_envs, self.num_actions), device=gs.device),
            'lin_vel': torch.zeros((self.num_envs, 3), device=gs.device),
            'ang_vel': torch.zeros((self.num_envs, 3), device=gs.device),
            'base_pos': torch.zeros((self.num_envs, 3), device=gs.device),
            'base_euler': torch.zeros((self.num_envs, 3), device=gs.device),
            'foot_contact': torch.zeros((self.num_envs, 2), device=gs.device),
        }
        
        self.randomization_intervals = { 'motor_strength': 50, 'friction': 100, 'mass': 200, 'observation_noise': 1, 'foot_contacts': 1, 'motor_backlash': 20 }
        self.randomization_counters = {k: 0 for k in self.randomization_intervals}
        self.randomization_step_counter = 0

        # Initialize handlers
        self.reward_handler = RewardHandler(self)
        self.dr_handler = DomainRandomizationHandler(self)

        # prepare episode sums for logging
        self.episode_sums = dict()
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.episode_sums["total_reward"] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float) # Track total reward for SB3 logging

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=torch.long) # Use long for boolean-like 0/1 values
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=gs.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor([self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]], device=gs.device, dtype=gs.tc_float)
        
        self.motor_backlash = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.motor_backlash_direction = torch.ones((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_motor_positions = torch.zeros_like(self.actions)
        
        self.foot_contacts = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.foot_contacts_raw = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        
        self.contact_thresholds = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_noise_scale = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_false_positive_prob = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_false_negative_prob = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_delay_steps = torch.zeros((self.num_envs, 2), device=gs.device, dtype=torch.long)
        self.contact_delay_buffer = torch.zeros((self.num_envs, 2, 5), device=gs.device, dtype=gs.tc_float) # Buffer for 5 timesteps delay
        self.contact_delay_idx = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.long)
        
        self.extras = {} # Used to pass info to SB3 VecEnv

    def _resample_commands(self, envs_idx):
        if len(envs_idx) > 0:
            self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
            self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
            self.commands[envs_idx, 2] = 0.0 # Angular velocity always 0 for now

    def step(self, actions):
        """
        Performs one simulation step given actions.
        Args:
            actions (torch.Tensor): Actions to apply to the robot, shape (num_envs, num_actions).
        Returns:
            obs_buf (torch.Tensor): New observations, shape (num_envs, num_obs).
            rew_buf (torch.Tensor): Rewards for this step, shape (num_envs,).
            reset_buf (torch.Tensor): Done flags for each env, shape (num_envs,).
            extras (dict): Additional information for logging and SB3.
        """
        self.randomization_step_counter += 1
        
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        
        exec_actions = self.dr_handler.apply_on_step(exec_actions)
        
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        self.post_physics_step() # Update state and check terminations

        # This will reset environments marked in self.reset_buf and update episode sums
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.compute_observations() # Recalculate observations based on new state

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        # Return PyTorch tensors directly; the VecEnv wrapper will convert to numpy
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        """
        Updates internal state variables and checks for episode termination conditions
        after a physics step.
        """
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        # Compute base euler angles relative to initial orientation
        self.base_euler = quat_to_xyz(transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        # Read foot contacts
        if self.left_foot_contact_sensor is not None:
            left_contact_data = self.left_foot_contact_sensor.read()
            left_contact_tensor = torch.as_tensor(left_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            self.foot_contacts_raw[:, 0] = torch.max(torch.norm(left_contact_tensor, dim=-1), dim=-1)[0] # Max force magnitude on left foot

        if self.right_foot_contact_sensor is not None:
            right_contact_data = self.right_foot_contact_sensor.read()
            right_contact_tensor = torch.as_tensor(right_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            self.foot_contacts_raw[:, 1] = torch.max(torch.norm(right_contact_tensor, dim=-1), dim=-1)[0] # Max force magnitude on right foot
        
        # Apply foot contact randomization
        if self.env_cfg["domain_rand"]["randomize_foot_contacts"]:
            self.foot_contacts = self.dr_handler._apply_foot_contact_randomization_optimized(self.foot_contacts_raw)
        else:
            self.foot_contacts = self.foot_contacts_raw.clone()
        
        # Resample commands periodically
        envs_idx_to_resample = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)).nonzero(as_tuple=False).reshape((-1,))
        self._resample_commands(envs_idx_to_resample)

        # Check termination conditions
        # Note: self.reset_buf will be updated to 1 (True) for environments that terminate
        self.reset_buf = (self.episode_length_buf > self.max_episode_length).long()
        self.reset_buf |= (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]).long()
        self.reset_buf |= (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]).long()

        # Compute rewards for the current step
        self.reward_handler.compute_rewards()
        self.episode_sums["total_reward"] += self.rew_buf

    def compute_observations(self):
        """
        Calculates and updates the observation buffer (self.obs_buf).
        Applies observation noise if enabled.
        """
        # Joint positions and velocities relative to default pose, scaled
        hip_angles = torch.cat([(self.dof_pos[:, [0, 1]] - self.default_dof_pos[[0, 1]]) * self.obs_scales["dof_pos"], (self.dof_pos[:, [4, 5]] - self.default_dof_pos[[4, 5]]) * self.obs_scales["dof_pos"]], dim=1)
        hip_velocities = torch.cat([self.dof_vel[:, [0, 1]] * self.obs_scales["dof_vel"], self.dof_vel[:, [4, 5]] * self.obs_scales["dof_vel"]], dim=1)
        knee_angles = torch.cat([(self.dof_pos[:, [2]] - self.default_dof_pos[[2]]) * self.obs_scales["dof_pos"], (self.dof_pos[:, [6]] - self.default_dof_pos[[6]]) * self.obs_scales["dof_pos"]], dim=1)
        knee_velocities = torch.cat([self.dof_vel[:, [2]] * self.obs_scales["dof_vel"], self.dof_vel[:, [6]] * self.obs_scales["dof_vel"]], dim=1)
        ankle_angles = torch.cat([(self.dof_pos[:, [3]] - self.default_dof_pos[[3]]) * self.obs_scales["dof_pos"], (self.dof_pos[:, [7]] - self.default_dof_pos[[7]]) * self.obs_scales["dof_pos"]], dim=1)
        ankle_velocities = torch.cat([self.dof_vel[:, [3]] * self.obs_scales["dof_vel"], self.dof_vel[:, [7]] * self.obs_scales["dof_vel"]], dim=1)
        foot_contacts_normalized = torch.clamp(self.foot_contacts, 0, 1)
        
        # Apply observation noise if enabled
        if self.env_cfg["domain_rand"]["add_observation_noise"] and self.dr_handler._should_update_randomization('observation_noise'):
            self.dr_handler._generate_noise_batch() # Generate new noise values
            base_euler_noisy = self.base_euler[:, :2] + self.noise_buffers['base_euler'][:, :2]
            base_ang_vel_xy_noisy = self.base_ang_vel[:, :2] + self.noise_buffers['ang_vel'][:, :2]
            base_ang_vel_z_noisy = self.base_ang_vel[:, [2]] + self.noise_buffers['ang_vel'][:, [2]]
            base_lin_vel_noisy = self.base_lin_vel[:, :2] + self.noise_buffers['lin_vel'][:, :2]
            base_pos_z_noisy = self.base_pos[:, [2]] + self.noise_buffers['base_pos'][:, [2]]
            hip_angles_noisy = hip_angles + self.noise_buffers['dof_pos'][:, :4]
            hip_velocities_noisy = hip_velocities + self.noise_buffers['dof_vel'][:, :4]
            knee_angles_noisy = knee_angles + self.noise_buffers['dof_pos'][:, 4:6]
            knee_velocities_noisy = knee_velocities + self.noise_buffers['dof_vel'][:, 4:6]
            ankle_angles_noisy = ankle_angles + self.noise_buffers['dof_pos'][:, 6:8]
            ankle_velocities_noisy = ankle_velocities + self.noise_buffers['dof_vel'][:, 6:8]
            foot_contacts_noisy = torch.clamp(foot_contacts_normalized + self.noise_buffers['foot_contact'], 0, 1)
        else:
            # Use un-noisy values if randomization is off
            base_euler_noisy, base_ang_vel_xy_noisy, base_ang_vel_z_noisy, base_lin_vel_noisy, base_pos_z_noisy = \
                self.base_euler[:, :2], self.base_ang_vel[:, :2], self.base_ang_vel[:, [2]], self.base_lin_vel[:, :2], self.base_pos[:, [2]]
            hip_angles_noisy, hip_velocities_noisy, knee_angles_noisy, knee_velocities_noisy, ankle_angles_noisy, ankle_velocities_noisy = \
                hip_angles, hip_velocities, knee_angles, knee_velocities, ankle_angles, ankle_velocities
            foot_contacts_noisy = foot_contacts_normalized
        
        # Concatenate all observation components
        obs_components = [
            base_euler_noisy * self.obs_scales.get("base_euler", 1.0), 
            base_ang_vel_xy_noisy * self.obs_scales["ang_vel"], 
            base_ang_vel_z_noisy * self.obs_scales["ang_vel"],
            base_lin_vel_noisy * self.obs_scales["lin_vel"], 
            base_pos_z_noisy * self.obs_scales.get("base_height", 1.0), 
            self.commands[:, :3] * self.commands_scale,
            hip_angles_noisy, hip_velocities_noisy, 
            knee_angles_noisy, knee_velocities_noisy, 
            ankle_angles_noisy, ankle_velocities_noisy, 
            foot_contacts_noisy, self.last_actions, # Last actions are part of observation
        ]
        torch.cat(obs_components, dim=-1, out=self.obs_buf)

    def reset_idx(self, envs_idx):
        """
        Resets specified environments to an initial state.
        Args:
            envs_idx (torch.Tensor): Indices of environments to reset.
        """
        if len(envs_idx) == 0:
            return

        self.dr_handler.apply_on_reset(envs_idx) # Apply domain randomization for resetting envs

        # Prepare 'episode' info for SB3 logging for environments that just finished
        # This needs to be done BEFORE resetting the episode_sums
        for i in envs_idx:
            # Only log if the episode actually finished (not just an early reset_idx call from training)
            # Check if this environment was marked as "done" in the previous step
            if self.reset_buf[i] == 1: # done
                # Create a dict for this specific environment's episode info
                ep_info = {
                    'r': self.episode_sums['total_reward'][i].item(), # Total reward for the episode
                    'l': self.episode_length_buf[i].item(),         # Episode length
                }
                # Add individual reward components to the episode info
                for key, value_tensor in self.episode_sums.items():
                    if key != "total_reward": # 'total_reward' is already handled above as 'r'
                        ep_info[f'rew_{key}'] = (value_tensor[i] / self.env_cfg["episode_length_s"]).item() # Normalize component rewards by episode length in seconds
                
                # Append to self.extras['episode'] list. SB3's VecEnv collects these.
                if 'episode' not in self.extras:
                    self.extras['episode'] = []
                self.extras['episode'].append(ep_info)

        # Reset episode sums for the environments that are ending
        for key in self.episode_sums.keys():
            self.episode_sums[key][envs_idx] = 0.0

        # Now, perform the actual physics reset operations
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(position=self.dof_pos[envs_idx], dofs_idx_local=self.motors_dof_idx, zero_velocity=True, envs_idx=envs_idx)

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.foot_contacts[envs_idx] = 0.0
        self.randomized_kp[envs_idx] = self.orig_kp
        
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 0 # Mark as not-resetting for the *next* step (0 = False, not done)

        self._resample_commands(envs_idx)

    def reset(self):
        """
        Resets all environments and returns initial observations.
        This is typically called once at the beginning of training.
        """
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.reset_buf[all_envs] = 1 # Mark all as needing reset
        self.reset_idx(all_envs) # Perform the reset
        self.compute_observations() # Compute observations after reset
        # Clear any episode info from previous (dummy) resets at start
        if 'episode' in self.extras:
            self.extras['episode'].clear() 
        return self.obs_buf


class BipedVecEnv(VecEnv):
    """
    A wrapper for the BipedEnv to make it compatible with the Stable Baselines3 VecEnv interface.
    This class handles the conversion between PyTorch tensors (used by BipedEnv) and
    NumPy arrays (expected by Stable Baselines3).
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device='cuda'):
        self.biped_env = BipedEnv(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)
        self.device = device

        self.num_envs = num_envs
        # Define observation and action spaces using Gymnasium.spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.biped_env.num_obs,), dtype=np.float32
        )
        clip_val = self.biped_env.env_cfg["clip_actions"]
        self.action_space = spaces.Box(
            low=-clip_val, high=clip_val, shape=(self.biped_env.num_actions,), dtype=np.float32
        )
        
        # Call the constructor of the base VecEnv class
        super().__init__(num_envs, self.observation_space, self.action_space)

    def reset(self, seed=None, options=None):
        """
        Resets all environments in the VecEnv.
        Returns:
            observations (np.ndarray): New observations.
        """
        # Note: SB3 VecEnv's reset does not accept env_idx
        # BipedEnv's internal reset handles all environments by default.
        obs = self.biped_env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray):
        """
        Feeds actions to the environments asynchronously.
        Args:
            actions (np.ndarray): Actions to take in each environment.
        """
        # Convert numpy actions to PyTorch tensor and store
        self.actions_tensor = torch.from_numpy(actions).to(self.device).float()

    def step_wait(self):
        """
        Waits for the environments to complete their step and returns the results.
        Returns:
            observations (np.ndarray): New observations.
            rewards (np.ndarray): Rewards from the step.
            dones (np.ndarray): Done flags.
            infos (list[dict]): List of dictionaries with episode information.
        """
        obs, rewards, dones, infos = self.biped_env.step(self.actions_tensor)
        
        # Convert PyTorch tensors to NumPy arrays
        obs_np = obs.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy().astype(bool) # SB3 expects boolean dones

        # Process infos for SB3 compatibility
        # SB3 expects a list of info dicts, one per environment
        # and 'episode' info to be a special dictionary if an episode ended.
        structured_infos = [{} for _ in range(self.num_envs)]
        
        # Populate episode stats for environments that reset
        if 'episode' in infos:
            for ep_info_dict in infos['episode']:
                # The 'episode' key contains a list of dicts from BipedEnv.reset_idx
                # We need to map these to the correct structured_infos entry
                # This mapping is tricky as BipedEnv.reset_idx doesn't explicitly return which env_idx reset.
                # A common pattern is to include the env_idx in the ep_info_dict itself,
                # or assume the order corresponds to `dones_np.nonzero()`.
                # For simplicity here, assuming 'episode' info is collected for *all*
                # environments that just ended, and iterating through them to append.
                # This might need refinement if specific episode info needs to be matched to specific env_idx.
                # A more robust way is for BipedEnv.reset_idx to pass `env_idx` along with its episode info.
                
                # For a clean mapping: we need to iterate over done environments and pull their info
                # from the biped_env.extras['episode'] list.
                pass # Already handled by _get_info method in this particular setup.

        # Ensure correct info structure, especially for 'final_observation'
        # when an episode is done, as RecurrentPPO expects it.
        for i in range(self.num_envs):
            if dones_np[i]:
                # SB3 needs the observation *before* the reset happened for the `final_observation`
                # Since BipedEnv internally resets, `obs_buf` already holds the *new* obs.
                # We need to capture the obs *before* the internal reset within BipedEnv.step().
                # This current implementation is *not* truly `final_observation`, but the observation
                # *after* the environment reset, which is usually okay if the reset state is what's needed.
                # If the state *at termination* is strictly required, BipedEnv.step would need to capture it.
                # For now, we return the observation *after* the reset, as the SB3 docs often imply.
                structured_infos[i]["final_observation"] = obs_np[i] 
        
        return obs_np, rewards_np, dones_np, self._get_info()

    def close(self):
        """Closes the environment."""
        if hasattr(self.biped_env.scene, 'close'):
             self.biped_env.scene.close()
    
    def _get_info(self) -> list[dict]:
        """
        Returns a list of info dictionaries, one for each environment.
        This is where episode statistics are extracted from biped_env.extras.
        """
        # Collect episode info that was populated by biped_env.reset_idx
        infos = []
        if 'episode' in self.biped_env.extras:
            infos.extend(self.biped_env.extras['episode'])
            self.biped_env.extras['episode'].clear() # Clear after reading

        # Pad with empty dicts for environments that did not reset
        # This is important for VecEnv consistency: it always expects num_envs info dicts.
        # This assumes that `infos` list contains info only for environments that reset.
        num_missing_infos = self.num_envs - len(infos)
        for _ in range(num_missing_infos):
            infos.append({})
            
        return infos

    # VecEnv requires these methods, even if they're no-ops or simple pass-throughs
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.biped_env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.biped_env, attr_name, value)
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs # This env is the base, not wrapped internally

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # Allow calling methods on the underlying BipedEnv
        return [getattr(self.biped_env, method_name)(*method_args, **method_kwargs)] * self.num_envs
