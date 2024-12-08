import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import wandb
import os
from tqdm import tqdm
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import copy
from torchvision.transforms import RandAugment

# -------------------------
# Utility Classes and Functions
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
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# -------------------------
# Vision Transformer Modules
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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x)  # (B, N, 3*inner_dim)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head)  # (B, N, 3, heads, dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, heads, N, dim_head)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, dim_head)
        out = out.transpose(1, 2).reshape(B, N, -1)  # (B, N, heads * dim_head)
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
                nn.LayerNorm(dim),
                Attention(dim, dim_head=dim_head, heads=heads, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # (B, num_patches, dim)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, dim)
        x = x + self.pos_embedding[:, :N + 1, :]  # (B, 1 + num_patches, dim)
        x = self.dropout(x)

        for norm1, attn, norm2, ff in self.transformer:
            x = x + attn(norm1(x))
            x = x + ff(norm2(x))

        x = x[:, 0]  # (B, dim)
        x = self.mlp_head(x)  # (B, num_classes)
        return x

# -------------------------
# Label Smoothing Cross Entropy
# -------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        """
        Args:
            eps (float): Smoothing factor.
            reduction (str): Reduction method ('mean' or 'sum').
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predictions (logits) from the model.
            target (Tensor): Ground truth labels.
        """
        c = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.eps / (c - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)

        if self.reduction == 'sum':
            return torch.sum(-true_dist * log_preds)
        else:
            return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))

# -------------------------
# Mixup Data Augmentation
# -------------------------
def mixup_data(x, y, alpha=0.2):
    """
    Applies Mixup augmentation to a batch of data.
    Args:
        x (Tensor): Input batch.
        y (Tensor): Labels.
        alpha (float): Mixup alpha parameter.
    Returns:
        Mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------
# Main
# -------------------------
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Initialize wandb
    wandb.init(project='vit-tiny-imagenet', config={
        'model': 'ViT',
        'dataset': 'Tiny ImageNet',
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'image_size': 64,
        'patch_size': 8,
        'dim': 512,
        'depth': 8,  # Increased depth for better representation
        'heads': 8,
        'mlp_dim': 2048,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'num_classes': 200,
        'label_smoothing': 0.1,
        'patience': 15,
        'mixup_alpha': 0.2,
        'randaugment_n': 2,
        'randaugment_m': 10,
        'scheduler': 'OneCycleLR',
        'gradient_clipping': 1.0,
        'use_mixup': True,
        'use_randaugment': True,
        'use_amp': True
    })
    config = wandb.config

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms for Tiny ImageNet
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        RandAugment(num_ops=config.randaugment_n, magnitude=config.randaugment_m) if config.use_randaugment else nn.Identity(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load Tiny ImageNet dataset
    train_dataset = datasets.ImageFolder(root='/home/sdanisetty/projects/cse527/data/tiny-imagenet-200/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(root='/home/sdanisetty/projects/cse527/data/tiny-imagenet-200/val', transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=8, persistent_workers=True)

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

    # Optionally, initialize weights (if not using pre-trained)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    # Loss function and optimizer
    criterion = LabelSmoothingCrossEntropy(eps=config.label_smoothing).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Learning Rate Scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        anneal_strategy='cos',
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    best_acc = 0.0
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch [{epoch+1}/{config.epochs}]')
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            if config.use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=config.mixup_alpha)
                targets_a, targets_b = targets_a.to(device), targets_b.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = model(inputs)
                if config.use_mixup:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)

            scaler.step(optimizer)
            scaler.update()

            if config.use_mixup:
                _, predicted = torch.max(outputs, 1)
                # For Mixup, accuracy isn't straightforward. Skipping accuracy logging per batch.
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

            running_loss += loss.item() * targets.size(0)

            # Update Learning Rate
            scheduler.step()

            if not config.use_mixup and batch_idx % 100 == 0:
                loop.set_postfix({
                    'Train Loss': running_loss / total if total > 0 else 0.0,
                    'Train Acc': 100. * correct / total if total > 0 else 0.0,
                    'LR': optimizer.param_groups[0]['lr']
                })

        # Validation
        model.eval()
        test_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation', leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                test_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        acc = 100. * correct_val / total_val
        test_loss_avg = test_loss / total_val

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': running_loss / len(train_loader.dataset),
            'train_acc': 100. * correct / total if not config.use_mixup else 'N/A',
            'test_loss': test_loss_avg,
            'test_acc': acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch [{epoch+1}/{config.epochs}] - Train Loss: {running_loss / len(train_loader.dataset):.4f} "
              f"- Train Acc: {100. * correct / total if not config.use_mixup else 'N/A'}% "
              f"- Val Loss: {test_loss_avg:.4f} - Val Acc: {acc:.2f}%")

        # Checkpointing best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_tinyimagenet_vit.pth')
            print(f"Saved Best Model with Accuracy: {best_acc:.2f}%")

        # Early Stopping check
        early_stopping(acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Restoring best model weights.")
            model.load_state_dict(early_stopping.best_model_weights)
            break

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")
    wandb.finish()

if __name__ == '__main__':
    main()
