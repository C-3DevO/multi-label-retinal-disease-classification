import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Per-class thresholds for [DR, Glaucoma, AMD]
THRESHOLDS = [0.40, 0.30, 0.45]


# ========================
# Reproducibility (optional but recommended)
# ========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic can be slower; set False if you want speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# Dataset preparation
# ========================
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
        labels = torch.tensor(row[1:].values.astype("float32"))
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


# ========================
# (Optional) losses you may want later
# NOTE: user previously said class-balanced/focal may be restricted.
# Keep them here for convenience; choose loss_type="bce" to comply.
# ========================
def compute_class_weights(csv_file):
    """
    Compute simple inverse-frequency weights for multi-label BCE:
    w_c = N / (C * n_c)
    """
    df = pd.read_csv(csv_file)
    labels = df.iloc[:, 1:].values  # shape (N, C)
    N = labels.shape[0]
    C = labels.shape[1]
    class_counts = labels.sum(axis=0)
    class_counts = np.clip(class_counts, 1.0, None)  # prevent div-by-zero
    class_weights = N / (C * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)


class ClassBalancedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        # Important: BCEWithLogitsLoss "weight" expects per-element weights;
        # broadcasting works if you shape weights as (C,) and targets are (B,C).
        self.register_buffer("w", class_weights)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        loss = self.bce(logits, targets)  # (B,C)
        loss = loss * self.w  # broadcast (C,) -> (B,C)
        return loss.mean()


# ========================
# Attention block
# ========================
class SpatialMHA(nn.Module):
    """
    Multi-Head Self-Attention over spatial feature maps
    x: (B,C,H,W) -> tokens: (B, H*W, C)
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        return tokens.transpose(1, 2).reshape(b, c, h, w)


# ========================
# Backbones + MHA
# ========================
class EfficientNetWithMHA(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, num_heads=4):
        super().__init__()
        base = models.efficientnet_b0(pretrained=pretrained)
        self.features = base.features  # output channels = 1280 for b0
        self.attn = SpatialMHA(embed_dim=1280, num_heads=num_heads)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class ResNet18WithMHA(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, num_heads=4):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.attn = SpatialMHA(embed_dim=512, num_heads=num_heads)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_model(backbone="efficientnet", num_classes=3, pretrained=True, use_mha=True, num_heads=4):
    if backbone == "resnet18":
        if use_mha:
            return ResNet18WithMHA(num_classes=num_classes, pretrained=pretrained, num_heads=num_heads)
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "efficientnet":
        if use_mha:
            return EfficientNetWithMHA(num_classes=num_classes, pretrained=pretrained, num_heads=num_heads)
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError("Unsupported backbone")


# ========================
# Freeze helpers
# ========================
def freeze_backbone(model, backbone: str):
    """
    Freeze everything except classifier head and attention.
    """
    for name, p in model.named_parameters():
        if backbone == "efficientnet":
            if not (name.startswith("classifier") or name.startswith("attn")):
                p.requires_grad = False
        elif backbone == "resnet18":
            if not (name.startswith("fc") or name.startswith("attn")):
                p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ========================
# Pretrained backbone loading (features only)
# ========================
def load_pretrained_features_only(model, ckpt_path: str):
    """
    Loads checkpoint weights while skipping classifier/fc head weights.
    Works for both efficientnet and resnet checkpoints (as long as key names match).
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")
    filtered = {k: v for k, v in state_dict.items() if not ("classifier" in k or "fc" in k)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("Loaded pretrained backbone (features only).")
    if len(unexpected) > 0:
        print("Unexpected keys (ignored):", unexpected[:10])
    if len(missing) > 0:
        # missing often includes classifier/fc keys which is fine
        print("Missing keys (ok):", missing[:10])


# ========================
# Train / Val loop
# ========================
def run_one_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n += bs

    return total_loss / max(n, 1)


# ========================
# Main training function: Phase 1 + Phase 2
# ========================
def train_one_backbone(
    backbone,
    train_csv, val_csv, test_csv,
    train_image_dir, val_image_dir, test_image_dir,
    epochs=20, batch_size=32, lr=1e-4, img_size=256,
    save_dir="checkpoints",
    pretrained_backbone=None,
    use_mha=True,
    loss_type="bce",   # "bce" or "class_balanced"
    phase1_epochs=6,
    phase1_lr=1e-4,
    phase2_lr=5e-5,
    num_heads=4,
    seed=42
):
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv,   val_image_dir,   transform)
    test_ds  = RetinaMultiLabelDataset(test_csv,  test_image_dir,  transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = build_model(backbone, num_classes=3, pretrained=True, use_mha=use_mha, num_heads=num_heads).to(device)

    # checkpoint path
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone features (optional)
    if pretrained_backbone is not None:
        load_pretrained_features_only(model, pretrained_backbone)

    # criterion
    if loss_type == "class_balanced":
        class_w = compute_class_weights(train_csv).to(device)
        print("Class weights:", class_w.detach().cpu().numpy())
        criterion = ClassBalancedBCELoss(class_w)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ========================
    # PHASE 1: Frozen backbone
    # ========================
    print("\n===== Phase 1: Frozen backbone (head + attn only) =====")
    freeze_backbone(model, backbone)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=phase1_lr)

    best_val_loss = float("inf")

    for epoch in range(phase1_epochs):
        train_loss = run_one_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        val_loss   = run_one_epoch(model, val_loader,   device, criterion, optimizer=None)

        print(f"[Phase 1] Epoch {epoch+1}/{phase1_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print("Saved best model (Phase 1)")

    # Load best from phase 1 before phase 2
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded best Phase 1 model for Phase 2")

    # ========================
    # PHASE 2: Full fine-tune
    # ========================
    print("\n===== Phase 2: Full fine-tuning =====")
    unfreeze_all(model)

    optimizer = optim.Adam(model.parameters(), lr=phase2_lr)

    for epoch in range(epochs):
        train_loss = run_one_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        val_loss   = run_one_epoch(model, val_loader,   device, criterion, optimizer=None)

        print(f"[Phase 2] Epoch {epoch+1}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print("Saved best model (overall)")

    # ========================
    # Testing
    # ========================
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > np.array(THRESHOLDS)).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    disease_names = ["DR", "Glaucoma", "AMD"]

    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, zero_division=0)
        recall = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"\n{disease} Results [{backbone}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")

    # ========================
    # Onsite inference (no labels)
    # ========================
    print("\nRunning onsite inference (no labels)...")

    onsite_csv = os.path.join(BASE_DIR, "onsite_test_submission.csv")
    onsite_image_dir = os.path.join(BASE_DIR, "images", "onsite_test")

    onsite_ds = RetinaOnsiteDataset(onsite_csv, onsite_image_dir, transform)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    onsite_predictions = []
    with torch.no_grad():
        for imgs, img_names in onsite_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > np.array(THRESHOLDS)).astype(int)

            for name, p in zip(img_names, preds):
                onsite_predictions.append([name, int(p[0]), int(p[1]), int(p[2])])

    submission = pd.DataFrame(onsite_predictions, columns=["id", "D", "G", "A"])
    submission_path = os.path.join(BASE_DIR, "onsite_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Onsite submission saved to: {submission_path}")


# ========================
# main
# ========================
if __name__ == "__main__":
    train_csv = os.path.join(BASE_DIR, "train.csv")
    val_csv   = os.path.join(BASE_DIR, "val.csv")
    test_csv  = os.path.join(BASE_DIR, "offsite_test.csv")

    train_image_dir = os.path.join(BASE_DIR, "images", "train")
    val_image_dir   = os.path.join(BASE_DIR, "images", "val")
    test_image_dir  = os.path.join(BASE_DIR, "images", "offsite_test")

    pretrained_backbone = os.path.join(BASE_DIR, "pretrained_backbone", "ckpt_efficientnet_ep50.pt")

    backbone = "efficientnet"  # ["resnet18", "efficientnet"]

    train_one_backbone(
        backbone=backbone,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        epochs=25,                 # Phase 2 epochs
        batch_size=32,
        lr=1e-4,                   # not used directly; kept for compatibility
        img_size=256,
        pretrained_backbone=pretrained_backbone,
        use_mha=True,
        loss_type="bce",           # change to "class_balanced" if allowed
        phase1_epochs=7,
        phase1_lr=1e-4,
        phase2_lr=1e-5,
        num_heads=4,
        seed=42
    )
