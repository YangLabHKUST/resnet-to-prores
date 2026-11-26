"""
Learning Rate Schedulers
"""

import math


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing Learning Rate Scheduler
    
    Uses linear warmup at the beginning of training, followed by cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, 
                 min_lr=0, last_epoch=-1):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            base_lr: Base learning rate
            min_lr: Minimum learning rate
            last_epoch: Index of the last epoch
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        
        # Initialize learning rate
        self.step()

    def get_lr(self):
        """Compute current learning rate"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (self.base_lr - self.min_lr) * alpha]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (self.base_lr - self.min_lr) * cosine_decay]

    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        lr = self.get_lr()[0]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class StepLRScheduler:
    """
    Step Learning Rate Scheduler
    Reduces learning rate at specified epochs
    """
    def __init__(self, optimizer, step_size, gamma=0.1, base_lr=0.1):
        """
        Args:
            optimizer: PyTorch optimizer
            step_size: Number of epochs between LR reductions
            gamma: Learning rate decay factor
            base_lr: Base learning rate
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = base_lr
        self.last_epoch = 0
        
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        
        lr = self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def get_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Scheduler type ('warmup_cosine', 'step')
        **kwargs: Scheduler-specific parameters
        
    Returns:
        Learning rate scheduler instance
    """
    if scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(optimizer, **kwargs)
    elif scheduler_type == 'step':
        return StepLRScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
