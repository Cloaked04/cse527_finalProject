import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import wandb
import os
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.utils.parametrize as parametrize
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Helper functions
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

# For use with parametrize
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

# Scaled dot product attention function
def scaled_dot_product_attention(q, k, v, dropout_p=0., training=True):
    d_k = q.size(-1)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = F.softmax(attn_weights, dim=-1)
    if training and dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    output = torch.matmul(attn_weights, v)
    return output

# Attention and FeedForward classes
class Attention(nn.Module):
    def __init__(self, dim, *, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_q = NormLinear(dim, dim_inner)
        self.to_k = NormLinear(dim, dim_inner)
        self.to_v = NormLinear(dim, dim_inner)

        self.dropout = dropout

        self.q_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))
        self.k_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))

        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # Query key rmsnorm
        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        out = scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout,
            training=self.training
        )

        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, *, dim_inner, dropout=0.):
        super().__init__()
        dim_inner = int(dim_inner * 2 / 3)

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = NormLinear(dim, dim_inner)
        self.to_gate = NormLinear(dim, dim_inner)

        self.hidden_scale = nn.Parameter(torch.ones(dim_inner))
        self.gate_scale = nn.Parameter(torch.ones(dim_inner))

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale
        gate = gate * self.gate_scale * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden

        hidden = self.dropout(hidden)
        return self.to_out(hidden)

# nViT base class
class nViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout=0.,
        channels=3,
        dim_head=64,
        residual_lerp_scale_init=None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)
        num_patches = patch_height_dim * patch_width_dim

        self.channels = channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.image_size = image_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size),
            NormLinear(patch_dim, dim, norm_dim_in=False),
        )

        self.abs_pos_emb = NormLinear(dim, num_patches)

        residual_lerp_scale_init = default(residual_lerp_scale_init, 1. / depth)

        self.dim = dim
        self.scale = dim ** 0.5

        self.layers = nn.ModuleList([])
        self.residual_lerp_scales = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, dim_inner=mlp_dim, dropout=dropout),
            ]))

            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
            ]))

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        device = x.device

        tokens = self.to_patch_embedding(x)

        seq_len = tokens.shape[-2]
        pos_emb = self.abs_pos_emb.weight[torch.arange(seq_len, device=device)]

        tokens = l2norm(tokens + pos_emb)

        for (attn, ff), residual_scales in zip(self.layers, self.residual_lerp_scales):
            attn_alpha, ff_alpha = residual_scales

            attn_out = l2norm(attn(tokens))
            tokens = l2norm(tokens.lerp(attn_out, attn_alpha * self.scale))

            ff_out = l2norm(ff(tokens))
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha * self.scale))

        # Classification token (mean pooling)
        tokens = tokens.mean(dim=1)
        logits = self.mlp_head(tokens)
        return logits

def main():
    # Initialize wandb
    wandb.init(project='nvit-cifar10', config={
        'model': 'nViT',
        'dataset': 'CIFAR-10',
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'image_size': 32,
        'patch_size': 4,
        'dim': 384,
        'depth': 6,
        'heads': 6,
        'mlp_dim': 384 * 4,
        'dropout': 0.1,
        'num_classes': 10,
        'dim_head': 64
    })
    config = wandb.config

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(config.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = nViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        dim_head=config.dim_head
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training loop
    best_acc = 0.0
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                wandb.log({
                    'train_loss': running_loss / (batch_idx + 1),
                    'train_acc': 100. * correct / total,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

        # Validation
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

        acc = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        wandb.log({
            'test_loss': avg_test_loss,
            'test_acc': acc,
            'epoch': epoch
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {100. * correct / total:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))

        scheduler.step()

    print(f"Training completed. Best Test Accuracy: {best_acc:.2f}%")
    wandb.finish()

if __name__ == '__main__':
    main()
