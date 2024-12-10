import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import wandb
import os
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange


class Attention(nn.Module):
    def __init__(self, dim, *, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, dim_inner, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Modified ViT for CIFAR-100 without normalization and with early stopping
class ViT(nn.Module):
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
        dropout=0.0,
        emb_dropout=0.0,
        channels=3,
        dim_head=64
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_size = patch_size
        self.dim = dim

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                Attention(dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, dim_inner=mlp_dim, dropout=dropout)
            ]))

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, dim)
        x = x + self.pos_embedding[:, :N + 1, :]
        x = self.dropout(x)

        for attn, ff in self.transformer:
            x = x + attn(x)
            x = x + ff(x)

        x = x[:, 0]
        x = self.mlp_head(x)
        return x


def main():
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Initialize wandb
    wandb.init(project='vit-cifar100-noNorm', config={
        'model': 'ViT',
        'dataset': 'CIFAR-100',
        'epochs': 100,
        'batch_size': 128,  # Batch size for CIFAR-100
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'image_size': 32,   # CIFAR-100 image size
        'patch_size': 4,    # Patch size for CIFAR-100
        'dim': 512,        # Model dimension
        'depth': 8,        # Transformer depth
        'heads': 8,        # Number of heads
        'mlp_dim': 512 * 4,   # MLP hidden dimension
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'num_classes': 100,
        'early_stopping_patience': 10  # Early stopping patience
    })
    config = wandb.config

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms for CIFAR-100 without normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Removed normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # Removed normalization
    ])

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = ViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Early Stopping parameters
    patience = config.early_stopping_patience
    epochs_no_improve = 0
    best_acc = 0.0
    best_epoch = 0

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

        # Print training progress
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

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
            'epoch': epoch + 1
        })

        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {acc:.2f}%")

        # Check for improvement
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_cifar100_vit.pth')
            print(f"New best model saved at epoch {epoch + 1} with accuracy {best_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement in accuracy for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        scheduler.step()

    print(f"Training completed. Best Test Accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    wandb.finish()


if __name__ == '__main__':
    main()
