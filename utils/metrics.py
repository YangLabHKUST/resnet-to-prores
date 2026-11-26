"""
Metrics calculation utilities for training and evaluation
"""

import torch
import time


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name='metric'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class AccuracyMeter:
    """Computes top-k accuracy"""
    def __init__(self, topk=(1, 5)):
        self.topk = topk
        self.reset()

    def reset(self):
        self.corrects = {k: 0 for k in self.topk}
        self.total = 0

    def update(self, output, target):
        """
        Args:
            output: Model output (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
        """
        batch_size = target.size(0)
        self.total += batch_size

        _, pred = output.topk(max(self.topk), 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            self.corrects[k] += correct_k.item()

    def get_accuracy(self, k=1):
        if self.total == 0:
            return 0.0
        return 100.0 * self.corrects[k] / self.total

    def __str__(self):
        acc_str = ', '.join([f'Top-{k}: {self.get_accuracy(k):.2f}%' 
                            for k in self.topk])
        return acc_str


def accuracy(output, target, topk=(1,)):
    """
    Computes top-k accuracy
    
    Args:
        output: Model output (batch_size, num_classes)
        target: Ground truth labels (batch_size,)
        topk: Tuple of k values to compute
        
    Returns:
        list: Accuracy for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Timer:
    """Timer for measuring elapsed time"""
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def reset(self):
        self.start_time = None
        self.end_time = None


def format_time(seconds):
    """Format time for display"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save checkpoint"""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device='cuda'):
    """Load checkpoint"""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    return epoch, best_acc
