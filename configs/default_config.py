"""
Default configuration for RDAUP model training
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone: str = "resnet50"  # ResNet backbone type
    pretrained: bool = True     # Use pretrained weights
    freeze_until: int = 1       # Freeze layers until this index
    feat_dim: int = 2048       # Feature dimension
    num_classes: int = 31      # Number of classes in Office-31

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Basic training params
    batch_size: int = 32
    epochs: int = 200
    lr: float = 0.001
    weight_decay: float = 5e-4
    momentum: float = 0.9
    
    # Loss weights
    lambda_adv: float = 1.0    # Adversarial loss weight
    lambda_ent: float = 1.0    # Entropy minimization weight
    lambda_upl: float = 1.0    # Uncertainty penalization weight
    
    # Optimization
    optimizer: str = "Adam"
    scheduler: str = "cosine"   # Learning rate scheduler
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Mixed precision training
    use_amp: bool = True
    grad_clip: float = 5.0

@dataclass
class DataConfig:
    """Dataset configuration"""
    # Data paths
    data_root: str = "./data/Office31"
    source_domain: str = "amazon"
    target_domain: str = "webcam"
    
    # Data augmentation
    random_horizontal_flip: bool = True
    random_rotation: int = 10
    color_jitter: bool = True
    
    # Normalization
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    # Input size
    image_size: int = 224
    num_workers: int = 4

@dataclass
class LoggingConfig:
    """Logging and checkpoint configuration"""
    # Directories
    log_dir: str = "experiments/logs"
    checkpoint_dir: str = "experiments/checkpoints"
    vis_dir: str = "experiments/visualizations"
    
    # Logging frequency
    log_interval: int = 10     # Steps between logging
    eval_interval: int = 1     # Epochs between evaluation
    save_interval: int = 10    # Epochs between checkpoints
    
    # Visualization
    tsne_plot: bool = True     # Generate t-SNE plots
    confusion_matrix: bool = True
    feature_visualization: bool = True
    
    # Wandb logging
    use_wandb: bool = False
    project_name: str = "RDAUP"
    entity: Optional[str] = None

@dataclass
class Config:
    """Master configuration"""
    # Random seed for reproducibility
    seed: int = 42
    
    # Sub-configurations
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.data.image_size > 0, "Image size must be positive"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert 0.0 <= self.training.lr <= 1.0, "Learning rate must be between 0 and 1"
        assert all(x >= 0 for x in [
            self.training.lambda_adv,
            self.training.lambda_ent,
            self.training.lambda_upl
        ]), "Loss weights must be non-negative"

# Default configuration
default_config = Config()
