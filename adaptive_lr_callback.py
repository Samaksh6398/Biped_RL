import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class ImprovedKLAdaptiveLRCallback(BaseCallback):
    def __init__(self, 
                 target_kl=0.02, 
                 lr_factor=0.8, 
                 patience=5,
                 smoothing_window=10,
                 min_lr=1e-6,
                 adaptation_threshold=0.8,
                 verbose=1):
        """
        Improved KL-based adaptive learning rate.
        
        Args:
            target_kl: Target KL divergence threshold
            lr_factor: Factor to multiply LR (< 1.0 for reduction, > 1.0 for increase)
            patience: Number of consecutive violations before adapting
            smoothing_window: Window size for smoothing KL values
            min_lr: Minimum learning rate
            adaptation_threshold: Fraction of target_kl to trigger LR increase
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.target_kl = target_kl
        self.lr_factor = lr_factor
        self.patience = patience
        self.min_lr = min_lr
        self.adaptation_threshold = adaptation_threshold
        
        # Smoothing for noisy KL values
        self.kl_history = deque(maxlen=smoothing_window)
        
        # Adaptation tracking
        self.violation_count = 0
        self.low_kl_count = 0
        self.adaptation_count = 0
        self.last_adaptation_step = 0
        
        # Statistics tracking
        self.kl_stats = {
            'mean': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0
        }
    
    def _on_step(self) -> bool:
        """Called after each step. Required by BaseCallback."""
        return True
    
    def _on_rollout_end(self) -> bool:
        """Called at the end of each rollout."""
        # Extract KL divergence from training logs
        current_kl = None
        
        # Try to get KL from different possible log keys
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            log_dict = self.model.logger.name_to_value
            current_kl = log_dict.get("train/approx_kl", None)
            
            # Fallback keys
            if current_kl is None:
                current_kl = log_dict.get("approx_kl", None)
            if current_kl is None:
                current_kl = log_dict.get("kl_divergence", None)
        
        if current_kl is None:
            return True  # Skip if KL not available
            
        # Add to history for smoothing
        self.kl_history.append(current_kl)
        
        # Calculate smoothed KL (reduces noise)
        smoothed_kl = np.mean(list(self.kl_history))
        
        # Update statistics
        self._update_kl_stats(smoothed_kl)
        
        # Make adaptation decision
        self._adapt_learning_rate(smoothed_kl, current_kl)
        
        return True
    
    def _update_kl_stats(self, kl_value):
        """Update KL statistics for monitoring."""
        if len(self.kl_history) > 1:
            kl_array = np.array(list(self.kl_history))
            self.kl_stats = {
                'mean': np.mean(kl_array),
                'std': np.std(kl_array),
                'max': np.max(kl_array),
                'min': np.min(kl_array)
            }
    
    def _adapt_learning_rate(self, smoothed_kl, raw_kl):
        """Adapt learning rate based on KL divergence."""
        current_lr = self.model.learning_rate
        adaptation_made = False
        
        # Check for KL violation (too high)
        if smoothed_kl > self.target_kl:
            self.violation_count += 1
            self.low_kl_count = 0  # Reset low KL counter
            
            # Reduce LR if consistent violations
            if self.violation_count >= self.patience and current_lr > self.min_lr:
                new_lr = max(current_lr * self.lr_factor, self.min_lr)
                self._update_learning_rate(new_lr, 'reduce', smoothed_kl, raw_kl)
                adaptation_made = True
                self.violation_count = 0
                
        # Check for very low KL (potential for LR increase)
        elif smoothed_kl < self.target_kl * self.adaptation_threshold:
            self.low_kl_count += 1
            self.violation_count = 0  # Reset violation counter
            
            # Increase LR if consistently low KL (conservative approach)
            if (self.low_kl_count >= self.patience * 2 and  # More conservative
                self.num_timesteps - self.last_adaptation_step > 50000):  # Time buffer
                
                new_lr = min(current_lr / self.lr_factor, 3e-4)  # Cap at reasonable value
                self._update_learning_rate(new_lr, 'increase', smoothed_kl, raw_kl)
                adaptation_made = True
                self.low_kl_count = 0
                
        else:
            # KL in acceptable range
            self.violation_count = 0
            self.low_kl_count = 0
        
        # Log current status
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            self._log_status(smoothed_kl, raw_kl, current_lr)
    
    def _update_learning_rate(self, new_lr, action, smoothed_kl, raw_kl):
        """Update the learning rate."""
        old_lr = self.model.learning_rate
        self.model.learning_rate = new_lr
        
        # Update optimizer learning rate
        if hasattr(self.model, 'policy'):
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        self.adaptation_count += 1
        self.last_adaptation_step = self.num_timesteps
        
        if self.verbose > 0:
            direction = "↓" if action == 'reduce' else "↑"
            print(f"Step {self.num_timesteps}: LR {action} {direction}")
            print(f"  {old_lr:.2e} → {new_lr:.2e}")
            print(f"  Smoothed KL: {smoothed_kl:.4f}, Raw KL: {raw_kl:.4f}")
            print(f"  Target KL: {self.target_kl:.4f}")
    
    def _log_status(self, smoothed_kl, raw_kl, current_lr):
        """Log current training status."""
        print(f"\n=== LR Adaptation Status (Step {self.num_timesteps}) ===")
        print(f"Current LR: {current_lr:.2e}")
        print(f"KL Divergence - Smoothed: {smoothed_kl:.4f}, Raw: {raw_kl:.4f}")
        print(f"KL Stats - Mean: {self.kl_stats['mean']:.4f}, "
              f"Std: {self.kl_stats['std']:.4f}")
        print(f"Adaptations made: {self.adaptation_count}")
        print(f"Violations: {self.violation_count}/{self.patience}, "
              f"Low KL: {self.low_kl_count}/{self.patience * 2}")
