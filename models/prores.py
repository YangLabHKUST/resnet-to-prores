import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet

def get_alpha(epoch, layer_idx, T):
    """
    Calculate alpha value for a specific layer.
    
    Args:
        epoch: Current epoch (0-based)
        layer_idx: Index of current layer (1 to total_layers)
        T: Hyperparameter T
    
    Returns:
        alpha: float in [0, 1]
    """
    if layer_idx <= 0: return 1.0
    
    # Formula: t / (T * l)
    # Adding small epsilon to avoid division by zero if T is 0 (though unlikely)
    denominator = T * layer_idx + 1e-6
    val = epoch / denominator
    
    return min(val, 1.0)

class PreNormBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-Norm: Norm before Conv
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        
        # Shortcut
        self.downsample = downsample
        self.stride = stride
        
        # Current epoch (updated by training script)
        self.current_epoch = 0
        
        # Hyperparameter T
        self.T = 1.0
    
    def set_epoch(self, epoch):
        """Called at the beginning of each epoch"""
        self.current_epoch = epoch
        
    def set_prores_params(self, T):
        """Set ProRes hyperparameters"""
        self.T = T
    
    def forward(self, x):
        identity = x
        
        # Pre-Norm: Norm -> Conv -> ReLU
        out = self.norm1(x)
        out = self.conv1(out)
        out = F.relu(out, inplace=True)
        
        out = self.norm2(out)
        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        
        out = self.norm3(out)
        out = self.conv3(out)
        
        # ProRes: Scale residual branch
        if self.layer_idx is not None:
            alpha = get_alpha(
                epoch=self.current_epoch,
                layer_idx=self.layer_idx,
                T=self.T
            )
            out = out * alpha
        
        # Shortcut handling
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out

class PostNormBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Post-Norm: Conv -> BN -> ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.current_epoch = 0
        self.T = 1.0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_prores_params(self, T):
        self.T = T
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # ProRes: Scale residual branch (after BN)
        if self.layer_idx is not None:
            alpha = get_alpha(
                epoch=self.current_epoch,
                layer_idx=self.layer_idx,
                T=self.T
            )
            out = out * alpha
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

def build_prores_resnet(block_type, layers, num_classes=1000, norm_type='pre', T=1.0, **kwargs):
    """
    Build ResNet with ProRes.
    
    Args:
        block_type: 'bottleneck'
        layers: List of block counts per stage
        norm_type: 'pre' or 'post'
        T: Hyperparameter T for alpha calculation
    """
    if norm_type == 'pre':
        block = PreNormBottleneck
    else:
        block = PostNormBottleneck
    
    model = ResNet(block, layers, num_classes=num_classes, **kwargs)
    
    # Assign global layer index to all residual blocks
    block_idx = 1
    
    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        stage = getattr(model, stage_name)
        for module in stage:
            if isinstance(module, (PreNormBottleneck, PostNormBottleneck)):
                module.layer_idx = block_idx
                module.set_prores_params(T)
                block_idx += 1
                
    return model

def prores_resnet50(num_classes=1000, norm_type='pre', T=1.0, **kwargs):
    return build_prores_resnet('bottleneck', [3, 4, 6, 3], num_classes=num_classes, norm_type=norm_type, T=T, **kwargs)

def prores_resnet101(num_classes=1000, norm_type='pre', T=1.0, **kwargs):
    return build_prores_resnet('bottleneck', [3, 4, 23, 3], num_classes=num_classes, norm_type=norm_type, T=T, **kwargs)

def prores_resnet152(num_classes=1000, norm_type='pre', T=1.0, **kwargs):
    return build_prores_resnet('bottleneck', [3, 8, 36, 3], num_classes=num_classes, norm_type=norm_type, T=T, **kwargs)
