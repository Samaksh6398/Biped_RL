import wandb
from rsl_rl.runners.wandb_logger import WandbLogger as RSLWandbLogger

class WandbLogger(RSLWandbLogger):
    """
    Custom Weights & Biases logger that integrates with rsl-rl.
    """
    def __init__(self, train_cfg, wandb_project, wandb_entity=None):
        """
        Initializes the WandbLogger.

        Args:
            train_cfg (dict): The training configuration dictionary.
            wandb_project (str): The name of the W&B project.
            wandb_entity (str, optional): The W&B entity (user or team). Defaults to None.
        """
        self.project = wandb_project
        self.entity = wandb_entity
        
        # Initialize W&B run
        wandb.init(
            project=self.project,
            entity=self.entity,
            config=train_cfg,
            reinit=True  # Allows re-initializing in the same process (e.g., notebooks)
        )
        
        print(f"W&B logging enabled. Project: '{self.project}', Entity: '{self.entity or 'default'}'.")
        
        # Call the parent class's init to set up the rest of the logger
        super().__init__(train_cfg)

    def finish(self):
        """
        Finishes the W&B run.
        """
        wandb.finish()