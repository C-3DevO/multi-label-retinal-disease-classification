# main.py 

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.transforms import RandAugment
from sklearn.metrics import precision_score, recall_score, f1_score
import timm
import warnings
warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = [0.35, 0.30, 0.40]
DISEASE_NAMES = ["DR", "Glaucoma", "AMD"]


# DATASETS 

class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:4].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels

class RetinaOnsiteDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name


# TRANSFORMS 

def get_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# LOSS AND CLASS WEIGHTS 

def compute_pos_weight(csv_file):
    df = pd.read_csv(csv_file)
    labels = df.iloc[:, 1:4].values
    N, C = labels.shape
    pos = labels.sum(axis=0) + 1e-6
    neg = N - pos + 1e-6
    pos_weight = neg / pos
    return torch.tensor(pos_weight, dtype=torch.float32)

class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()

def make_weighted_sampler(csv_file):
    df = pd.read_csv(csv_file)
    labels = df.iloc[:, 1:4].values
    label_counts = labels.sum(axis=0) + 1e-6
    class_weights = 1.0 / label_counts
    sample_weights = (labels * class_weights).sum(axis=1)
    sample_weights = sample_weights / sample_weights.sum()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ATTENTION MODULES 

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.conv2.out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se(out)
        return out

class MHABlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out + x_flat)
        return attn_out.permute(0, 2, 1).view(B, C, H, W)


# MODEL BUILDING 

def build_model(backbone="resnet18", num_classes=3, pretrained=True, attention=None):
    if backbone == "resnet18":
        if attention == "SE":
            model = ResNet(block=SEBasicBlock, layers=[2, 2, 2, 2])
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif attention == "MHA":
            model = models.resnet18(pretrained=pretrained)
            model.layer4.add_module("mha", MHABlock(512))
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "swin":
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
    elif backbone == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


# THRESHOLD SEARCH 

def find_best_thresholds(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            probs = torch.sigmoid(model(imgs)).cpu()
            all_probs.append(probs.numpy())
            all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    best_th = []
    ts = np.linspace(0.1, 0.9, 17)
    for i in range(all_labels.shape[1]):
        f1s = [f1_score(all_labels[:, i], (all_probs[:, i] > t).astype(int), zero_division=0) for t in ts]
        best_th.append(float(ts[np.argmax(f1s)]))
    return best_th


# GRADUAL UNFREEZING 

def gradual_unfreeze(model, backbone, epoch, total_epochs):
    if backbone not in ["resnet18", "efficientnet"]:
        return
    if epoch == int(0.25 * total_epochs):
        for name, p in model.named_parameters():
            if backbone == "resnet18":
                if ("layer4" in name or "fc" in name) and not p.requires_grad:
                    p.requires_grad = True
            elif backbone == "efficientnet":
                if ("features.6" in name or "features.7" in name or "classifier" in name) and not p.requires_grad:
                    p.requires_grad = True
    if epoch == int(0.5 * total_epochs):
        for name, p in model.named_parameters():
            if backbone == "resnet18":
                if ("layer3" in name or "layer4" in name) and not p.requires_grad:
                    p.requires_grad = True
            elif backbone == "efficientnet":
                if ("features.4" in name or "features.5" in name) and not p.requires_grad:
                    p.requires_grad = True


# TRAINING FUNCTION

def train_one_backbone(
    backbone="resnet18",
    train_csv="", val_csv="", test_csv="",
    train_image_dir="", val_image_dir="", test_image_dir="",
    epochs=40, batch_size=64, lr=1e-4, img_size=256,
    save_dir="checkpoints", pretrained_backbone=None,
    mode="full_finetune", loss_type="bce",
    focal_gamma=2.0,
    attention=None,
    use_sampler=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    num_workers = 0
    torch.backends.cudnn.benchmark = True
    
    print(f"\n Training {backbone} | Mode: {mode} | Loss: {loss_type} | Attention: {attention} | B={batch_size} | Workers={num_workers}")

    if backbone in ["swin", "vit"]:
        img_size = 224

    train_tf, val_tf = get_transforms(img_size)

    
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, train_tf)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, val_tf)
    test_ds = RetinaMultiLabelDataset(test_csv, test_image_dir, val_tf)

    
    if use_sampler:
        sampler = make_weighted_sampler(train_csv)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, 
                                num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)

    # Model
    model = build_model(backbone, num_classes=3, pretrained=True, attention=attention).to(device)

    if pretrained_backbone and os.path.exists(pretrained_backbone):
        model.load_state_dict(torch.load(pretrained_backbone, map_location=device), strict=False)
        print("Loaded external pretrained weights")

    if mode == "freeze_backbone":
        for name, p in model.named_parameters():
            if backbone == "resnet18":
                if "fc" not in name:
                    p.requires_grad = False
            elif backbone == "efficientnet":
                if "classifier" not in name:
                    p.requires_grad = False
            elif backbone in ["swin", "vit"]:
                if "head" not in name:
                    p.requires_grad = False

    # Loss
    if loss_type == "focal":
        pos_weight = compute_pos_weight(train_csv).to(device)
        criterion = FocalBCEWithLogits(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # AMP SCALER + Optimizer
    scaler = torch.cuda.amp.GradScaler()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if backbone in ["swin", "vit"]:
        optimizer = optim.AdamW(params, lr=lr, weight_decay=0.01)
    else:
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop with AMP
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}_{attention or 'none'}_{loss_type}_{mode}.pt")

    for epoch in range(epochs):
        if mode == "full_finetune":
            gradual_unfreeze(model, backbone, epoch, epochs)

        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # Load model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # Tune thresholds
    global THRESHOLDS
    THRESHOLDS = find_best_thresholds(model, val_loader, device)
    print(f"Best thresholds: {THRESHOLDS}")

    # Offsite Test
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            preds = (probs > np.array(THRESHOLDS)[None, :]).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n>>> OFFSITE TEST RESULTS ({backbone})")
    avg_f1 = 0.0
    for i, disease in enumerate(DISEASE_NAMES):
        yt, yp = y_true[:, i], y_pred[:, i]
        f1 = f1_score(yt, yp, zero_division=0)
        avg_f1 += f1
        print(f"{disease}: P={precision_score(yt, yp, zero_division=0):.4f}, "
              f"R={recall_score(yt, yp, zero_division=0):.4f}, F1={f1:.4f}")
    avg_f1 /= 3
    print(f"Average F1: {avg_f1:.4f}")

    # Onsite Inference
    onsite_csv = os.path.join(BASE_DIR, "onsite_test_submission.csv")
    onsite_image_dir = os.path.join(BASE_DIR, "images", "onsite_test")
    onsite_tf = val_tf
    onsite_ds = RetinaOnsiteDataset(onsite_csv, onsite_image_dir, onsite_tf)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    preds_list = []
    with torch.no_grad():
        for imgs, names in onsite_loader:
            imgs = imgs.to(device, non_blocking=True)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            preds = (probs > np.array(THRESHOLDS)[None, :]).astype(int)
            for name, p in zip(names, preds):
                preds_list.append([name, int(p[0]), int(p[1]), int(p[2])])

    submission = pd.DataFrame(preds_list, columns=["id", "D", "G", "A"])
    submission_path = os.path.join(BASE_DIR, f"onsite_submission_{backbone}_{attention or 'none'}_{loss_type}_{mode}.csv")
    submission.to_csv(submission_path, index=False)
    print(f"nsite submission saved: {submission_path}")

    return avg_f1


# MAIN

if __name__ == "__main__":
    train_csv = os.path.join(BASE_DIR, "train.csv")
    val_csv = os.path.join(BASE_DIR, "val.csv")
    test_csv = os.path.join(BASE_DIR, "offsite_test.csv")
    train_img = os.path.join(BASE_DIR, "images", "train")
    val_img = os.path.join(BASE_DIR, "images", "val")
    test_img = os.path.join(BASE_DIR, "images", "offsite_test")

    print(f"Working directory: {BASE_DIR}")

    # Task 1
    print("\n=== TASK 1: Transfer Learning ===")
    for backbone in ["resnet18", "efficientnet"]:
        for mode in ["freeze_backbone", "full_finetune"]:
            train_one_backbone(
                backbone=backbone,
                train_csv=train_csv, val_csv=val_csv, test_csv=test_csv,
                train_image_dir=train_img, val_image_dir=val_img, test_image_dir=test_img,
                epochs=40 if mode == "full_finetune" else 25,
                batch_size=64, lr=3e-4,
                mode=mode, loss_type="bce",
                attention=None
            )

    # Task 2
    print("\n=== TASK 2: Advanced Loss Functions ===")
    best_backbone = "efficientnet"
    for loss in ["focal"]:
        train_one_backbone(
            backbone=best_backbone,
            train_csv=train_csv, val_csv=val_csv, test_csv=test_csv,
            train_image_dir=train_img, val_image_dir=val_img, test_image_dir=test_img,
            epochs=40, batch_size=64, lr=3e-4,
            mode="full_finetune", loss_type=loss,
            attention=None
        )

    # Task 3
    print("\n=== TASK 3: Attention Mechanisms ===")
    for attn in ["SE", "MHA"]:
        train_one_backbone(
            backbone="resnet18",
            train_csv=train_csv, val_csv=val_csv, test_csv=test_csv,
            train_image_dir=train_img, val_image_dir=val_img, test_image_dir=test_img,
            epochs=40, batch_size=64, lr=3e-4,
            mode="full_finetune", loss_type="focal",
            attention=attn
        )

    # Task 4
    print("\n=== TASK 4: Transformer Backbones ===")
    for backbone in ["swin", "vit"]:
        train_one_backbone(
            backbone=backbone,
            train_csv=train_csv, val_csv=val_csv, test_csv=test_csv,
            train_image_dir=train_img, val_image_dir=val_img, test_image_dir=test_img,
            epochs=40, batch_size=32, lr=3e-5,
            mode="full_finetune", loss_type="focal",
            attention=None
        )

    print("\n ALL TASKS COMPLETE! )