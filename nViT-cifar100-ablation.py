import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import wandb
import os
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.nn.utils.parametrize as parametrize
import math
import random
import copy

####################
# Helper Functions #
####################

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim, p=2)

#############################
# Parametrization Classes  #
#############################

# L2Norm for parametrize
class L2Norm(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        return l2norm(t, dim=self.dim)

class NormLinear(nn.Module):
    def __init__(self, dim, dim_out, norm_dim_in=True):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=False)
        parametrize.register_parametrization(
            self.linear,
            'weight',
            L2Norm(dim=-1 if norm_dim_in else 0)
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

####################
# ConvStem Module  #
####################

class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

###############################
# Scaled Dot Product Attention#
###############################

def scaled_dot_product_attention(q, k, v, dropout_p=0., training=True):
    d_k = q.size(-1)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = F.softmax(attn_weights, dim=-1)
    if training and dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    output = torch.matmul(attn_weights, v)
    return output

class Attention(nn.Module):
    def __init__(self, dim, *, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_q = NormLinear(dim, dim_inner)
        self.to_k = NormLinear(dim, dim_inner)
        self.to_v = NormLinear(dim, dim_inner)
        self.dropout = dropout
        self.q_scale = nn.Parameter(torch.ones(heads,1,dim_head)*(dim_head**0.25))
        self.k_scale = nn.Parameter(torch.ones(heads,1,dim_head)*(dim_head**0.25))
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

    def forward(self,x):
        q,k,v = self.to_q(x), self.to_k(x), self.to_v(x)
        q,k,v = map(self.split_heads, (q,k,v))
        q,k = map(l2norm,(q,k))
        q = q*self.q_scale
        k = k*self.k_scale
        out = scaled_dot_product_attention(q,k,v, dropout_p=self.dropout, training=self.training)
        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, *, dim_inner, dropout=0.):
        super().__init__()
        dim_inner = int(dim_inner*2/3)
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.to_hidden = NormLinear(dim, dim_inner)
        self.to_gate = NormLinear(dim, dim_inner)
        self.hidden_scale = nn.Parameter(torch.ones(dim_inner))
        self.gate_scale = nn.Parameter(torch.ones(dim_inner))
        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

    def forward(self,x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)
        hidden = hidden*self.hidden_scale
        gate = gate*self.gate_scale*(self.dim**0.5)
        hidden = F.silu(gate)*hidden
        hidden = self.dropout(hidden)
        return self.to_out(hidden)

###################################
# nViT Model with Optional ConvStem#
###################################

class nViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0., channels=3, dim_head=64, residual_lerp_scale_init=None, use_convstem=False):
        super().__init__()
        assert divisible_by(image_size, patch_size), "Image must be divisible by patch_size"
        num_patches = (image_size//patch_size)*(image_size//patch_size)
        self.use_convstem = use_convstem
        self.dim = dim
        if use_convstem:
            # With ConvStem
            self.conv_stem = ConvStem(channels, dim)
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c h w -> b (h w) c'),
                NormLinear(dim,dim,norm_dim_in=False),
            )
        else:
            # Without ConvStem
            patch_dim = channels*(patch_size**2)
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2)-> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size),
                NormLinear(patch_dim, dim, norm_dim_in=False),
            )
        self.abs_pos_emb = NormLinear(dim, num_patches)
        residual_lerp_scale_init = default(residual_lerp_scale_init, 1./depth)
        self.scale = dim**0.5

        self.layers = nn.ModuleList([])
        self.residual_lerp_scales = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, dim_inner=mlp_dim, dropout=dropout)
            ]))
            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(dim)*residual_lerp_scale_init/self.scale),
                nn.Parameter(torch.ones(dim)*residual_lerp_scale_init/self.scale),
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self,x):
        device = x.device
        if self.use_convstem:
            x = self.conv_stem(x) # shape: B, dim, H/4, W/4 if patch_size=4
            tokens = self.to_patch_embedding(x)
        else:
            tokens = self.to_patch_embedding(x)

        seq_len = tokens.shape[-2]
        pos_emb = self.abs_pos_emb.weight[torch.arange(seq_len,device=device)]
        tokens = l2norm(tokens+pos_emb)
        for (attn, ff), residual_scales in zip(self.layers, self.residual_lerp_scales):
            attn_alpha, ff_alpha = residual_scales
            attn_out = l2norm(attn(tokens))
            tokens = l2norm(tokens.lerp(attn_out, attn_alpha*self.scale))
            ff_out = l2norm(ff(tokens))
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha*self.scale))
        tokens = tokens.mean(dim=1)
        return self.mlp_head(tokens)

#####################
# MixUp & CutMix    #
#####################

def mixup_data(x, y, alpha=0.2):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the MixUp loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

###############################
# Experiment Runner Function  #
###############################

def run_experiment(config):
    """
    Runs a single experiment based on the provided configuration.
    Logs metrics and model checkpoints to wandb.
    """
    # Construct run name based on configuration
    run_name = f"nViT_CIFAR100_convstem={config['use_convstem']}_mixup={config['use_mixup']}_aug={config['augment_level']}"
    wandb.init(project='nvit-cifar100-ablation', config=config, name=run_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning Experiment: {run_name} on device: {device}\n")

    # Data augmentation based on config
    if config['augment_level'] == 'baseline':
        # Minimal augmentations
        train_transform = transforms.Compose([
            transforms.RandomCrop(config['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    elif config['augment_level'] == 'advanced':
        # Advanced augmentations
        train_transform = transforms.Compose([
            transforms.RandomCrop(config['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    else:
        # Default to baseline if unknown
        train_transform = transforms.Compose([
            transforms.RandomCrop(config['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=4, pin_memory=True)

    # Initialize model
    model = nViT(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        dropout=config['dropout'],
        dim_head=config['dim_head'],
        use_convstem=config['use_convstem']
    ).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Early Stopping Parameters
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = config['patience']
    trigger_times = 0

    # Training Loop
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Apply MixUp if enabled
            if config['use_mixup']:
                inputs, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if config['use_mixup']:
                # Approximate correct predictions with MixUp
                correct_preds = lam * predicted.eq(y_a).sum().item() + (1 - lam) * predicted.eq(y_b).sum().item()
                correct += correct_preds
            else:
                correct += predicted.eq(targets).sum().item()

            # Logging intermediate batch metrics
            if batch_idx % 100 == 0:
                wandb.log({
                    'train_loss': running_loss / (batch_idx + 1),
                    'train_acc': 100. * correct / total,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

        # Epoch-wise Training Metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation Phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Epoch-wise Validation Metrics
        acc = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        wandb.log({
            'epoch': epoch,
            'test_loss': avg_test_loss,
            'test_acc': acc
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {acc:.2f}%")

        # Early Stopping Logic
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), f'best_nvit_{run_name}.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered!")
                break

        # Step the scheduler
        scheduler.step()

        # Log best accuracy so far
        wandb.log({'best_test_acc': best_acc})

    # Load Best Model Weights
    model.load_state_dict(best_model_wts)
    print(f"Training completed. Best Test Accuracy: {best_acc:.2f}%")
    wandb.finish()

#############################
# Experiment Configurations #
#############################

if __name__ == '__main__':
    # Set seeds for reproducibility
    set_seed(42)

    # Define the list of experiment configurations
    experiments = [
        # 1. Advanced Augmentations without ConvStem, no MixUp
        {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'image_size': 32,
            'patch_size': 1,
            'dim': 512,
            'depth': 8,
            'heads': 8,
            'mlp_dim': 512 * 4,
            'dropout': 0.1,
            'num_classes': 100,
            'dim_head': 64,
            'patience': 20,
            'use_convstem': False,
            'use_mixup': False,
            'augment_level': 'advanced'  # Advanced augmentations
        },

        # 2. MixUp only (baseline augmentations, no ConvStem)
        {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'image_size': 32,
            'patch_size': 1,
            'dim': 512,
            'depth': 8,
            'heads': 8,
            'mlp_dim': 512 * 4,
            'dropout': 0.1,
            'num_classes': 100,
            'dim_head': 64,
            'patience': 20,
            'use_convstem': False,
            'use_mixup': True,
            'augment_level': 'baseline'  # Baseline augmentations
        },

        # 3. ConvStem only (baseline augmentations, no MixUp)
        {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'image_size': 32,
            'patch_size': 1,
            'dim': 512,
            'depth': 8,
            'heads': 8,
            'mlp_dim': 512 * 4,
            'dropout': 0.1,
            'num_classes': 100,
            'dim_head': 64,
            'patience': 20,
            'use_convstem': True,
            'use_mixup': False,
            'augment_level': 'baseline'  # Baseline augmentations
        },

        # 4. ConvStem + MixUp (baseline augmentations)
        {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'image_size': 32,
            'patch_size': 1,
            'dim': 512,
            'depth': 8,
            'heads': 8,
            'mlp_dim': 512 * 4,
            'dropout': 0.1,
            'num_classes': 100,
            'dim_head': 64,
            'patience': 20,
            'use_convstem': True,
            'use_mixup': True,
            'augment_level': 'baseline'  # Baseline augmentations
        },

        # 5. ConvStem + Advanced Augmentations (no MixUp)
        {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'image_size': 32,
            'patch_size': 1,
            'dim': 512,
            'depth': 8,
            'heads': 8,
            'mlp_dim': 512 * 4,
            'dropout': 0.1,
            'num_classes': 100,
            'dim_head': 64,
            'patience': 20,
            'use_convstem': True,
            'use_mixup': False,
            'augment_level': 'advanced'  # Advanced augmentations
        },
    ]

    # Optionally, remove any experiments you have already run to prevent duplication
    # For example, if you have already run ConvStem + Advanced Augmentations + MixUp,
    # ensure it's not in the list. Adjust the experiments list accordingly.

    # Run all experiments sequentially
    for idx, exp_cfg in enumerate(experiments):
        print(f"\nStarting Experiment {idx+1}/{len(experiments)}")
        run_experiment(exp_cfg)
