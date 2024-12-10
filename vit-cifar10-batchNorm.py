import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import wandb
import os
import numpy as np
import random
from einops import rearrange
from einops.layers.torch import Rearrange
import copy

# -------------------------
# Helper Function
# -------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# -------------------------
# Vision Transformer Modules with BatchNorm
# -------------------------
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
        
        # BatchNorm after projection
        self.batch_norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x)  # (B, N, 3*inner_dim)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head)  # (B, N, 3, H, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, N, D)

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, -1)  # (B, N, H*D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Apply BatchNorm
        out = out.permute(0, 2, 1)  # (B, C, N)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)  # (B, N, C)
        
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # BatchNorm after FeedForward
        self.batch_norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = self.net(x)  # (B, N, dim)
        
        # Apply BatchNorm
        B, N, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, N)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # (B, N, C)
        
        return x

# -------------------------
# Modified ViT for CIFAR-10 with Batch Normalization and Early Stopping
# -------------------------
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
                FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

        # Retained LayerNorm before MLP head for classification stability
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # Optional: Remove if you want no normalization at all
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # (B, N, dim)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, dim)
        x = x + self.pos_embedding[:, :N + 1, :]
        x = self.dropout(x)

        for attn, ff in self.transformer:
            x = x + attn(x)  # Residual connection
            x = x + ff(x)    # Residual connection

        x = x[:, 0]  # (B, dim)
        x = self.mlp_head(x)  # (B, num_classes)
        return x

# -------------------------
# Early Stopping Utility
# -------------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation accuracy improved.
            verbose (bool): If True, prints a message for each validation accuracy improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_acc = 0.0
        self.best_model_weights = None

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"Initial best validation accuracy: {self.best_score:.2f}%")
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Validation accuracy improved from {self.best_score:.2f}% to {score:.2f}%")
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

# -------------------------
# Main Training Function
# -------------------------
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize wandb
    wandb.init(project='vit-cifar10-withBatchNorm', config={
        'model': 'ViT',
        'dataset': 'CIFAR-10',
        'epochs': 100,
        'batch_size': 128,  # Batch size for CIFAR-10
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'image_size': 32,   # CIFAR-10 image size
        'patch_size': 4,    # Patch size for CIFAR-10
        'dim': 384,         # Model dimension
        'depth': 6,         # Transformer depth
        'heads': 6,         # Number of heads
        'mlp_dim': 384 * 4, # MLP hidden dimension
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'num_classes': 10,
        'patience': 10      # Patience for early stopping
    })
    config = wandb.config

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms with Batch Normalization (standardization)
    transform_train = transforms.Compose([
        transforms.RandomCrop(config.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)

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
        emb_dropout=config.emb_dropout,
        channels=3,
        dim_head=64
    ).to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Early Stopping Parameters
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)

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

        # Calculate average training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation Phase
        model.eval()
        test_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        # Calculate average test loss and accuracy
        acc = 100. * correct_val / total_val
        avg_test_loss = test_loss / len(test_loader)
        wandb.log({
            'test_loss': avg_test_loss,
            'test_acc': acc,
            'epoch': epoch + 1
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {acc:.2f}%")

        # Early Stopping Check
        early_stopping(acc, model)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_cifar10_vit.pth')
            print(f"New best model saved at epoch {epoch + 1} with accuracy {best_acc:.2f}%")
        else:
            print(f"No improvement in accuracy for epoch {epoch + 1}")

        if early_stopping.early_stop:
            print(f"Early stopping triggered after epoch {epoch + 1}.")
            break

        # Step the scheduler
        scheduler.step()

    print(f"Training completed. Best Test Accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    wandb.finish()

if __name__ == '__main__':
    main()