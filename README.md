# Biped Robot RL Training

A reinforcement learning framework for training bipedal robots using Genesis Physics Engine and Stable-Baselines3 with advanced reward systems.

## 🏗️ Technology Stack

- **🔬 Physics Simulation**: Genesis Physics Engine (v0.3.1) - High-performance GPU-accelerated physics
- **🧠 RL Framework**: Stable-Baselines3 + SB3-Contrib (RecurrentPPO with LSTM)
- **🎯 Environment**: Custom Gymnasium-compatible vectorized environment
- **📊 Logging**: WandB + TensorBoard integration
- **⚡ Performance**: CUDA-accelerated training with mixed precision support
- **🎛️ Features**: Adaptive learning rate, advanced reward systems, checkpointing

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate the genesis environment
conda activate genesis

# OR create from scratch
conda env create -f environment_minimal.yml
bash install_genesis.sh
```

### 2. Basic Training

```bash
# Simple training run
python biped_train.py

# Training with WandB logging
python biped_train.py --wandb --wandb_project "my-biped-project"

# Training with more environments for faster learning
python biped_train.py --num_envs 2048 --wandb
```

## 📋 Command Line Arguments

### Core Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--exp_name` / `-e` | str | `"biped-walking-sb3-recurrentppo"` | Experiment name for logs |
| `--num_envs` / `-B` | int | `1024` | Number of parallel environments |
| `--total_timesteps` | int | `200000000` | Total training timesteps (200M) |

### Model Management

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--load_model` | str | `None` | Path to saved model to resume training |
| `--continue_training` | flag | `False` | Auto-load latest model from logs |

### WandB Integration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--wandb` | flag | `False` | Enable Weights & Biases logging |
| `--wandb_project` | str | `"biped-rl-sb3"` | WandB project name |
| `--wandb_entity` | str | `None` | WandB entity (username/team) |
| `--wandb_run_name` | str | `None` | Custom run name |

### Advanced Features

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--adaptive_lr` | flag | `False` | Enable KL-divergence based adaptive learning rate |

## 🎯 Training Examples

### Development Training
```bash
# Quick test with fewer environments
python biped_train.py --num_envs 64 --total_timesteps 1000000
```

### Production Training
```bash
# Full training with logging and adaptive LR
python biped_train.py \
    --num_envs 2048 \
    --total_timesteps 200000000 \
    --wandb \
    --wandb_project "biped-production" \
    --wandb_run_name "v1.0-full-training" \
    --adaptive_lr
```

### Resume Training
```bash
# Continue from latest checkpoint
python biped_train.py --continue_training --wandb

# Load specific model
python biped_train.py \
    --load_model "logs/biped-walking-sb3-recurrentppo/biped_model_10000000_steps.zip" \
    --wandb
```

### Hyperparameter Tuning
```bash
# Training with different configurations
python biped_train.py \
    --exp_name "biped-tuning-lr1e3" \
    --num_envs 1024 \
    --wandb \
    --wandb_run_name "lr-1e3-experiment"
```

## 🏆 Advanced Reward Systems

This framework includes sophisticated reward functions:

- **🚶 Foot Alternation Rewards**: Encourages natural gait patterns
- **📏 Height Termination**: Episodes end if robot falls below 0.30m
- **🌊 Sinusoidal Motion Rewards**: Promotes rhythmic, stable walking
- **⚖️ Balance & Stability**: Multi-component reward for robust locomotion

## 📊 Monitoring & Logging

### TensorBoard (Local)
```bash
tensorboard --logdir logs/
```

### WandB Dashboard
- Automatic logging of training metrics
- Real-time reward decomposition
- Model checkpoints and artifacts
- Hyperparameter tracking

## 🔧 Configuration

Key configuration files:
- `biped_config.py` - Environment, reward, and training parameters
- `environment_minimal.yml` - Conda environment specification
- `biped_env.py` - Custom Gymnasium environment wrapper

### Model Architecture
- **Policy**: MlpLstmPolicy (Actor-Critic with LSTM)
- **Network**: [512, 256, 128] → LSTM(512, 2 layers) → Actions/Values
- **Algorithm**: RecurrentPPO with experience replay
- **Activation**: ELU

### Training Hyperparameters
- **Learning Rate**: 1e-4 (adaptive available)
- **Batch Size**: `(num_envs × n_steps) ÷ 4`
- **PPO Epochs**: 10
- **Clip Range**: 0.2
- **GAE Lambda**: 0.95

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce number of environments
   python biped_train.py --num_envs 512
   ```

2. **Genesis Import Error**
   ```bash
   # Ensure proper environment activation
   conda activate genesis
   ```

3. **TensorBoard Missing**
   ```bash
   # Install tensorboard in environment
   conda run -n genesis_test pip install tensorboard==2.20.0
   ```

### Performance Tips

- **GPU Memory**: Use fewer environments if running out of VRAM
- **Training Speed**: Increase `--num_envs` for faster learning
- **Stability**: Enable `--adaptive_lr` for better convergence
- **Debugging**: Use smaller `--total_timesteps` for quick tests

## 📁 Output Structure

```
logs/
└── biped-walking-sb3-recurrentppo/
    ├── biped_model_*_steps.zip    # Model checkpoints
    ├── model_final.zip            # Final trained model
    ├── cfgs.pkl                   # Environment configurations
    └── RecurrentPPO_1/            # TensorBoard logs
```

