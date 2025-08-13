import torch

def compute_observations(env):
    """
    Computes the observation tensor from the environment's state, applying scaling and noise.
    """
    # Extract and scale joint angles and velocities
    hip_angles = torch.cat([
        (env.dof_pos[:, [0, 1]] - env.default_dof_pos[[0, 1]]) * env.obs_scales["dof_pos"],
        (env.dof_pos[:, [4, 5]] - env.default_dof_pos[[4, 5]]) * env.obs_scales["dof_pos"],
    ], dim=1)
    hip_velocities = torch.cat([
        env.dof_vel[:, [0, 1]] * env.obs_scales["dof_vel"],
        env.dof_vel[:, [4, 5]] * env.obs_scales["dof_vel"],
    ], dim=1)
    knee_angles = torch.cat([
        (env.dof_pos[:, [2]] - env.default_dof_pos[[2]]) * env.obs_scales["dof_pos"],
        (env.dof_pos[:, [6]] - env.default_dof_pos[[6]]) * env.obs_scales["dof_pos"],
    ], dim=1)
    knee_velocities = torch.cat([
        env.dof_vel[:, [2]] * env.obs_scales["dof_vel"],
        env.dof_vel[:, [6]] * env.obs_scales["dof_vel"],
    ], dim=1)
    ankle_angles = torch.cat([
        (env.dof_pos[:, [3]] - env.default_dof_pos[[3]]) * env.obs_scales["dof_pos"],
        (env.dof_pos[:, [7]] - env.default_dof_pos[[7]]) * env.obs_scales["dof_pos"],
    ], dim=1)
    ankle_velocities = torch.cat([
        env.dof_vel[:, [3]] * env.obs_scales["dof_vel"],
        env.dof_vel[:, [7]] * env.obs_scales["dof_vel"],
    ], dim=1)
    foot_contacts_normalized = torch.clamp(env.foot_contacts, 0, 1)

    # Apply observation noise if enabled
    if env.env_cfg["domain_rand"]["add_observation_noise"] and env.dr_manager._should_update_randomization('observation_noise'):
        env.dr_manager._generate_noise_batch()
        base_euler_noisy = env.base_euler[:, :2] + env.dr_manager.noise_buffers['base_euler'][:, :2]
        base_ang_vel_xy_noisy = env.base_ang_vel[:, :2] + env.dr_manager.noise_buffers['ang_vel'][:, :2]
        base_ang_vel_z_noisy = env.base_ang_vel[:, [2]] + env.dr_manager.noise_buffers['ang_vel'][:, [2]]
        base_lin_vel_noisy = env.base_lin_vel[:, :2] + env.dr_manager.noise_buffers['lin_vel'][:, :2]
        base_pos_z_noisy = env.base_pos[:, [2]] + env.dr_manager.noise_buffers['base_pos'][:, [2]]
        hip_angles_noisy = hip_angles + env.dr_manager.noise_buffers['dof_pos'][:, :4]
        hip_velocities_noisy = hip_velocities + env.dr_manager.noise_buffers['dof_vel'][:, :4]
        knee_angles_noisy = knee_angles + env.dr_manager.noise_buffers['dof_pos'][:, 4:6]
        knee_velocities_noisy = knee_velocities + env.dr_manager.noise_buffers['dof_vel'][:, 4:6]
        ankle_angles_noisy = ankle_angles + env.dr_manager.noise_buffers['dof_pos'][:, 6:8]
        ankle_velocities_noisy = ankle_velocities + env.dr_manager.noise_buffers['dof_vel'][:, 6:8]
        foot_contacts_noisy = torch.clamp(foot_contacts_normalized + env.dr_manager.noise_buffers['foot_contact'], 0, 1)
    else:
        # Use clean observations if noise is disabled
        base_euler_noisy = env.base_euler[:, :2]
        base_ang_vel_xy_noisy = env.base_ang_vel[:, :2]
        base_ang_vel_z_noisy = env.base_ang_vel[:, [2]]
        base_lin_vel_noisy = env.base_lin_vel[:, :2]
        base_pos_z_noisy = env.base_pos[:, [2]]
        hip_angles_noisy = hip_angles
        hip_velocities_noisy = hip_velocities
        knee_angles_noisy = knee_angles
        knee_velocities_noisy = knee_velocities
        ankle_angles_noisy = ankle_angles
        ankle_velocities_noisy = ankle_velocities
        foot_contacts_noisy = foot_contacts_normalized

    # Concatenate all components into the final observation buffer
    obs_components = [
        base_euler_noisy * env.obs_scales.get("base_euler", 1.0),
        base_ang_vel_xy_noisy * env.obs_scales["ang_vel"],
        base_ang_vel_z_noisy * env.obs_scales["ang_vel"],
        base_lin_vel_noisy * env.obs_scales["lin_vel"],
        base_pos_z_noisy * env.obs_scales.get("base_height", 1.0),
        env.commands[:, :3] * env.commands_scale,
        hip_angles_noisy,
        hip_velocities_noisy,
        knee_angles_noisy,
        knee_velocities_noisy,
        ankle_angles_noisy,
        ankle_velocities_noisy,
        foot_contacts_noisy,
        env.last_actions,
    ]
    
    torch.cat(obs_components, dim=-1, out=env.obs_buf)
    return env.obs_buf