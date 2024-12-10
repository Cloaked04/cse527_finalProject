import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import wandb
import os
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from PIL import Image
from torchvision.transforms import functional as F
import pandas as pd
from pathlib import Path

# Helper function
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Attention and FeedForward classes
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class OxfordPetsSegmentation(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Setup paths
        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'annotations' / 'trimaps'
        
        # Get file lists
        self.images = sorted(list(self.images_dir.glob('*.jpg')))
        if split == 'train':
            self.images = self.images[:int(len(self.images)*0.8)]
        else:
            self.images = self.images[int(len(self.images)*0.8):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + '.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Process mask
        mask = np.array(mask) - 1  # mask values are {0,1,2}
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class ViTSegmentation(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        num_classes,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

        self.decoder = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                     h=image_height//patch_height, 
                     w=image_width//patch_width, 
                     p1=patch_height, 
                     p2=patch_width)
        )

        # Final convolution layers for segmentation
        self.final_layers = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for norm1, attn, norm2, ff in self.transformer:
            x = x + attn(norm1(x))
            x = x + ff(norm2(x))

        # Remove cls token for reconstruction
        x = x[:, 1:]
        
        # Decode to image space
        x = self.decoder(x)
        
        # Final convolution layers
        x = self.final_layers(x)
        
        return x

def train_model():
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Initialize wandb
    wandb.init(project='vit-oxford-pets-segmentation', config={
        'model': 'ViT-Segmentation',
        'dataset': 'Oxford-Pets',
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'image_size': 224,
        'patch_size': 16,
        'dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'num_classes': 3  # Background, Foreground, Boundary
    })
    config = wandb.config

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = OxfordPetsSegmentation(
        root_dir='path/to/oxford-iiit-pet',  # Replace with actual path
        split='train',
        transform=train_transform
    )
    
    val_dataset = OxfordPetsSegmentation(
        root_dir='path/to/oxford-iiit-pet',  # Replace with actual path
        split='val',
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = ViTSegmentation(
        image_size=config.image_size,
        patch_size=config.patch_size,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        num_classes=config.num_classes,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Metrics
    def calculate_iou(pred, target, num_classes=3):
        pred = pred.argmax(dim=1)  # For multi-class predictions
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
    best_iou = 0.0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        total_iou = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks).item()

            if batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_iou': calculate_iou(outputs, masks).item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

        avg_train_loss = total_loss / len(train_loader)
        avg_train_iou = total_iou / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                val_loss += criterion(outputs, masks).item()
                val_iou += calculate_iou(outputs, masks).item()

                # Log sample predictions
                if batch_idx % 100 == 0:
                    pred_masks = outputs.argmax(dim=1).cpu().numpy()
                    true_masks = masks.cpu().numpy()
                    images_np = images.cpu().numpy()

                    wandb.log({
                        "predictions": [
                            wandb.Image(images_np[i].transpose(1,2,0), caption=f"Prediction",
                                masks={
                                    "predictions": {"mask_data": pred_masks[i]},
                                    "ground_truth": {"mask_data": true_masks[i]}
                                }
                            ) for i in range(min(4, images_np.shape[0]))
                        ]
                    })

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        wandb.log({
            'epoch': epoch,
            'val_loss': avg_val_loss,
            'val_iou': avg_val_iou
        })

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), 'best_oxford_pets_segmentation.pth')

        scheduler.step()

        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

    print(f"Training completed. Best validation IoU: {best_iou:.4f}")
    wandb.finish()

if __name__ == '__main__':
    train_model()

