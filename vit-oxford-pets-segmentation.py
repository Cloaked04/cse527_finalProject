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
import glob
from torchvision.transforms import functional as F
from torchvision.io import read_image
import pandas as pd
from pathlib import Path

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
        
        # Convert mask to binary (foreground/background)
        mask = np.array(mask)
        mask = (mask == 2).astype(np.float32)  # 2 is foreground
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = F.to_tensor(mask)
        
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
        self.patch_to_embedding = nn.Sequential(
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
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

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
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
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
        'emb_dropout': 0.1
    })
    config = wandb.config

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = OxfordPetsSegmentation(
        root_dir='path/to/oxford-iiit-pet',
        split='train',
        transform=transform
    )
    
    val_dataset = OxfordPetsSegmentation(
        root_dir='path/to/oxford-iiit-pet',
        split='val',
        transform=transform
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
        dropout=config.dropout,
        emb_dropout=config.emb_dropout
    ).to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Metrics
    def calculate_iou(pred, target):
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + 1e-6) / (union + 1e-6)

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
            total_iou += calculate_iou(outputs, masks)

            if batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_iou': calculate_iou(outputs, masks),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                val_loss += criterion(outputs, masks).item()
                val_iou += calculate_iou(outputs, masks)

                # Log sample predictions
                if batch_idx % 100 == 0:
                    wandb.log({
                        "predictions": wandb.Image(
                            images[0],
                            masks={
                                "predictions": {"mask_data": outputs[0].cpu().numpy()},
                                "ground_truth": {"mask_data": masks[0].cpu().numpy()}
                            }
                        )
                    })

        avg_val_iou = val_iou / len(val_loader)
        wandb.log({
            'epoch': epoch,
            'val_loss': val_loss / len(val_loader),
            'val_iou': avg_val_iou
        })

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), 'best_oxford_pets_segmentation.pth')

        scheduler.step()

    wandb.finish()

if __name__ == '__main__':
    train_model()
