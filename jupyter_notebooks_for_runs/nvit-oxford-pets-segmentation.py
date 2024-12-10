import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import wandb
import os
import numpy as np
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.utils.parametrize as parametrize

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

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.,
            scale=1.
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
        self.residual_lerp_scales = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, dim_inner=mlp_dim, dropout=dropout),
            ]))

            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
            ]))

    def forward(self, x):
        device = x.device

        tokens = self.to_patch_embedding(x)

        seq_len = tokens.shape[-2]
        pos_emb = self.abs_pos_emb.weight[torch.arange(seq_len, device=device)]

        tokens = l2norm(tokens + pos_emb)

        for (attn, ff), (attn_alpha, ff_alpha) in zip(self.layers, self.residual_lerp_scales):
            attn_out = l2norm(attn(tokens))
            tokens = l2norm(tokens.lerp(attn_out, attn_alpha * self.scale))

            ff_out = l2norm(ff(tokens))
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha * self.scale))

        return tokens

# Segmentation-specific decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SegmentationViT(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        num_classes=3,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        dim_head=64
    ):
        super().__init__()
        
        # Initialize encoder
        self.encoder = nViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=dim,  # Using dim as embedding size
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            channels=in_channels,
            dim_head=dim_head
        )
        
        # Decoder layers
        self.decoder = nn.ModuleList([
            DecoderBlock(dim, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            nn.ConvTranspose2d(128, num_classes, kernel_size=2, stride=2)
        ])
        
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)  # B, N, D
        
        # Reshape features to spatial form
        h = w = int(self.encoder.image_size // self.encoder.patch_size)
        features = features.reshape(features.shape[0], h, w, -1).permute(0, 3, 1, 2)
        
        # Decoder
        for decoder_layer in self.decoder:
            features = decoder_layer(features)
        
        return features

# Initialize wandb
wandb.init(project='nvit-oxford-pets', config={
    'model': 'SegmentationViT',
    'dataset': 'Oxford-IIIT Pet',
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'image_size': 256,
    'patch_size': 16,
    'dim': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    'dropout': 0.1,
    'num_classes': 3,  # Background, Border, Pet
    'dim_head': 64
})
config = wandb.config

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

target_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation',
                             transform=transform, target_transform=target_transform, download=True)
test_dataset = OxfordIIITPet(root='./data', split='test', target_types='segmentation',
                            transform=transform, target_transform=target_transform, download=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model
model = SegmentationViT(
    image_size=config.image_size,
    patch_size=config.patch_size,
    in_channels=3,
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

# Training metrics
def calculate_iou(pred, target, num_classes=3):
    pred = pred.argmax(dim=1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union == 0:
            iou_per_class.append(1.0 if intersection == 0 else 0.0)
        else:
            iou_per_class.append((intersection + 1e-6) / (union + 1e-6))
    return sum(iou_per_class) / len(iou_per_class)

# Training loop
best_iou = 0
scaler = torch.cuda.amp.GradScaler()

for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    total_iou = 0
    num_train_batches = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        masks = masks.squeeze(1).long()  # Remove channel dimension
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_iou += calculate_iou(outputs.detach(), masks).item()
        num_train_batches += 1
        
        if batch_idx % 10 == 0:
            wandb.log({
                'batch_train_loss': loss.item(),
                'batch_train_iou': calculate_iou(outputs.detach(), masks).item(),
            })

    avg_train_loss = total_loss / num_train_batches
    avg_train_iou = total_iou / num_train_batches

    # Validation
    model.eval()
    val_loss = 0
    val_iou = 0
    num_val_batches = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1).long()

            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            val_iou += calculate_iou(outputs.detach(), masks).item()
            num_val_batches += 1

    avg_val_loss = val_loss / num_val_batches
    avg_val_iou = val_iou / num_val_batches

    # Log epoch metrics
    wandb.log({
        'train_loss': avg_train_loss,
        'train_iou': avg_train_iou,
        'val_loss': avg_val_loss,
        'val_iou': avg_val_iou,
        'epoch': epoch,
        'learning_rate': optimizer.param_groups[0]['lr']
    })

    # Save the best model
    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))

    print(f"Epoch {epoch + 1}/{config.epochs} - "
          f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

    scheduler.step()

print(f"Training completed. Best validation IoU: {best_iou:.4f}")
wandb.finish()