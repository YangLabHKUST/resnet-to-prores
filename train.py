"""
Training script for ResNet50 on ImageNet100
Configured via configs/configs.yaml only.
"""

import os
import yaml
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Import models
from models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet18_prenorm,
    resnet34_prenorm,
    resnet50_prenorm,
    resnet101_prenorm,
    resnet152_prenorm,
    prores_resnet50,
    prores_resnet101,
    prores_resnet152,
)
from data import get_dataloaders
from utils import (AverageMeter, AccuracyMeter, Timer, format_time,
                   save_checkpoint, load_checkpoint, get_scheduler)


def load_config(config_path='configs/configs.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                    device, epoch, config):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter('Loss')
    acc_meter = AccuracyMeter(topk=(1, 5))
    timer = Timer()
    timer.start()
    
    # Update ProRes epoch info
    if 'model' in config and 'prores' in config['model'] and config['model']['prores'].get('enabled', False):
        for module in model.modules():
            if hasattr(module, 'set_epoch'):
                module.set_epoch(epoch - 1)
    
    print_freq = config['logging'].get('print_freq', 100)
    total_epochs = config['training']['epochs']

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        
        if config['amp'].get('enabled', False):
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Update statistics
        losses.update(loss.item(), images.size(0))
        acc_meter.update(outputs, targets)
        
        # Print progress
        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(train_loader):
            elapsed = timer.elapsed()
            print(f'Epoch [{epoch}/{total_epochs}] '
                  f'Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} '
                  f'{acc_meter} '
                  f'Time: {format_time(elapsed)}')
    
    timer.stop()
    return losses.avg, acc_meter.get_accuracy(1), acc_meter.get_accuracy(5)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    
    losses = AverageMeter('Loss')
    acc_meter = AccuracyMeter(topk=(1, 5))
    
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        losses.update(loss.item(), images.size(0))
        acc_meter.update(outputs, targets)
    
    return losses.avg, acc_meter.get_accuracy(1), acc_meter.get_accuracy(5)


def main():
    # Load configuration
    print('Loading configuration from configs/configs.yaml...')
    config = load_config()
    
    # Handle simple CLI overrides (e.g. --amp) manually
    import sys
    for arg in sys.argv:
        if arg == '--amp':
            print("CLI override: Enabling AMP")
            if 'amp' not in config: config['amp'] = {}
            config['amp']['enabled'] = True
    
    # Set random seed
    seed = config['misc'].get('seed', 42)
    set_seed(seed)
    
    # Set directories
    work_dir = config['experiment'].get('work_dir', './work_dirs')
    job_name = config['experiment'].get('job_name', 'default_job')
    
    # Create timestamped subdirectory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    mode = 'val' if config['eval'].get('eval_only', False) else 'train'
    sub_dir = f"{mode}_{timestamp}"
    
    # Base output directory
    output_dir = os.path.join(work_dir, job_name, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy config file to output directory
    shutil.copy('configs/configs.yaml', os.path.join(output_dir, 'configs.yaml'))
    
    # Checkpoint directory
    save_dir_cfg = config['checkpoint'].get('save_dir', 'checkpoints')
    if not os.path.isabs(save_dir_cfg):
        save_dir = os.path.join(output_dir, save_dir_cfg)
    else:
        save_dir = save_dir_cfg
    os.makedirs(save_dir, exist_ok=True)
    
    print(f'Experiment: {job_name}')
    print(f'Output directory: {output_dir}')
    print(f'Save directory: {save_dir}')
    
    # Setup logging
    import sys
    log_dir_cfg = config['logging'].get('log_dir', 'logs')
    if not os.path.isabs(log_dir_cfg):
        log_dir = os.path.join(output_dir, log_dir_cfg)
    else:
        log_dir = log_dir_cfg
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'train.log')
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    # Set device
    device_name = config['misc'].get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    data_cfg = config['data']
    print(f'Loading data from {data_cfg["root"]}...')
    train_loader, val_loader, num_classes = get_dataloaders(
        data_root=data_cfg['root'],
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 4),
        img_size=data_cfg.get('img_size', 224),
        pin_memory=data_cfg.get('pin_memory', True),
        dataset_name=data_cfg.get('dataset', 'imagenet100')
    )
    print(f'Number of classes: {num_classes}')
    
    # Create model
    model_cfg = config['model']
    model_name = model_cfg['name']
    print(f'Creating model: {model_name}...')
    norm_type = model_cfg.get('norm_type', 'post')
    if norm_type not in ('pre', 'post'):
        raise ValueError(f"Unknown norm_type: {norm_type}")

    # Determine if we should use CIFAR stem
    dataset_name = data_cfg.get('dataset', 'imagenet100')
    img_size = data_cfg.get('img_size', 224)
    use_cifar_stem = (dataset_name.lower() in ('cifar10', 'cifar100') and img_size <= 64)
    
    if use_cifar_stem:
        print(f"Using CIFAR stem (3x3 conv, no maxpool) for small images ({img_size}x{img_size})")

    prores_cfg = model_cfg.get('prores', {})
    use_prores = prores_cfg.get('enabled', False)

    if use_prores:
        T = prores_cfg.get('T', 1.0)
        print(f'Using ProRes with {norm_type}-norm, T={T}')

        model_fn = {
            'resnet50': prores_resnet50,
            'resnet101': prores_resnet101,
            'resnet152': prores_resnet152
        }.get(model_name)

        if model_fn is None:
            raise ValueError(f"ProRes not implemented for {model_name}")

        model = model_fn(num_classes=num_classes, norm_type=norm_type, T=T, cifar_stem=use_cifar_stem)
    else:
        if norm_type == 'post':
            model_fn = {
                'resnet18': resnet18,
                'resnet34': resnet34,
                'resnet50': resnet50,
                'resnet101': resnet101,
                'resnet152': resnet152
            }[model_name]
        else:
            model_fn = {
                'resnet18': resnet18_prenorm,
                'resnet34': resnet34_prenorm,
                'resnet50': resnet50_prenorm,
                'resnet101': resnet101_prenorm,
                'resnet152': resnet152_prenorm
            }[model_name]

        print(f'Using standard ResNet with {norm_type}-norm (ProRes disabled)')
        model = model_fn(num_classes=num_classes, cifar_stem=use_cifar_stem)

        
    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss function
    crit_cfg = config['training']['criterion']
    criterion = nn.CrossEntropyLoss(label_smoothing=crit_cfg.get('label_smoothing', 0.1))
    
    # Optimizer
    opt_cfg = config['training']['optimizer']
    opt_type = opt_cfg.get('type', 'SGD')
    lr = opt_cfg.get('lr', 0.1)
    weight_decay = opt_cfg.get('weight_decay', 1e-4)
    momentum = opt_cfg.get('momentum', 0.9)
    
    print(f'Using optimizer: {opt_type}')
    if opt_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif opt_type == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.999)),
            weight_decay=weight_decay
        )
    elif opt_type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.999)),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    # Learning rate scheduler
    sched_cfg = config['training']['scheduler']
    sched_type = sched_cfg.get('type', 'warmup_cosine')
    
    if sched_type == 'warmup_cosine':
        scheduler = get_scheduler(
            optimizer, 'warmup_cosine',
            warmup_epochs=sched_cfg.get('warmup_epochs', 5),
            total_epochs=config['training']['epochs'],
            base_lr=lr,
            min_lr=sched_cfg.get('min_lr', 0.0)
        )
    else:
        scheduler = get_scheduler(
            optimizer, 'step',
            step_size=30,
            gamma=0.1,
            base_lr=lr
        )
    
    # Mixed precision training
    scaler = GradScaler() if config['amp'].get('enabled', False) else None
    
    # Resume training
    start_epoch = 0
    best_acc1 = 0.0
    resume_path = config['checkpoint'].get('resume', '')
    
    if resume_path and os.path.isfile(resume_path):
        print(f'Loading checkpoint from {resume_path}...')
        start_epoch, best_acc1 = load_checkpoint(
            resume_path, model, optimizer, device
        )
        print(f'Resumed from epoch {start_epoch}, best_acc1: {best_acc1:.2f}%')
    
    # Evaluation only mode
    if config['eval'].get('eval_only', False):
        print('Evaluating...')
        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Top-1 Accuracy: {val_acc1:.2f}%')
        print(f'Validation Top-5 Accuracy: {val_acc5:.2f}%')
        return
    
    # Training loop
    print('\n' + '='*60)
    print('Start training')
    print('='*60)
    
    total_epochs = config['training']['epochs']
    eval_freq = config['eval'].get('eval_freq', 1)
    save_freq = config['checkpoint'].get('save_freq', 10)
    
    total_timer = Timer()
    total_timer.start()
    
    for epoch in range(start_epoch, total_epochs):
        print(f'\nEpoch: {epoch+1}/{total_epochs}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Training
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch+1, config
        )
        
        print(f'Train Loss: {train_loss:.4f}, '
              f'Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%')
        
        # Evaluation
        if (epoch + 1) % eval_freq == 0:
            val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device)
            print(f'Val Loss: {val_loss:.4f}, '
                  f'Top-1: {val_acc1:.2f}%, Top-5: {val_acc5:.2f}%')
            
            # Save best model
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)
            
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc1,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f'Saved best model with accuracy: {best_acc1:.2f}%')
        
        # Periodic checkpoint saving
        if (epoch + 1) % save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc1,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Update learning rate
        scheduler.step()
    
    total_timer.stop()
    print('\n' + '='*60)
    print(f'Training completed in {format_time(total_timer.elapsed())}')
    print(f'Best Top-1 Accuracy: {best_acc1:.2f}%')
    print('='*60)


if __name__ == '__main__':
    main()
