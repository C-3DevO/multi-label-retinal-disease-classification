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

# Class thresholds for [DR, Glaucoma, AMD]
THRESHOLDS = [0.40, 0.30, 0.45]



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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
    
class EfficientNetWithSE(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        base = models.efficientnet_b0(pretrained=pretrained)
        self.features = base.features         
        self.se = SEBlock(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

class ResNet18WithSE(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4           

        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

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
    class_counts = np.clip(class_counts, 1.0, None)  
    class_weights = N / (C * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)


class ClassBalancedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.register_buffer("w", class_weights)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        loss = self.bce(logits, targets)  
        loss = loss * self.w  
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


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
        tokens = x.flatten(2).transpose(1, 2)  

        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        return tokens.transpose(1, 2).reshape(b, c, h, w)



class EfficientNetWithMHA(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, num_heads=4):
        super().__init__()
        base = models.efficientnet_b0(pretrained=pretrained)
        self.features = base.features  
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


def build_model(backbone="efficientnet", num_classes=3, pretrained=True,
                use_mha=False, use_se=False, num_heads=4):

    if backbone == "resnet18":
        if use_mha:
            return ResNet18WithMHA(num_classes, pretrained, num_heads)
        if use_se:
            return ResNet18WithSE(num_classes, pretrained)

        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "efficientnet":
        if use_mha:
            return EfficientNetWithMHA(num_classes, pretrained, num_heads)
        if use_se:
            return EfficientNetWithSE(num_classes, pretrained)

        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model


    raise ValueError("Unsupported backbone")




def freeze_backbone(model, backbone: str):
    for name, p in model.named_parameters():
        if backbone == "efficientnet":
            if not (name.startswith("classifier") or name.startswith("attn") or name.startswith("se")):
                p.requires_grad = False
        elif backbone == "resnet18":
            if not (name.startswith("fc") or name.startswith("attn") or name.startswith("se")):
                p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True



def load_pretrained_features_only(model, ckpt_path: str): 
    state_dict = torch.load(ckpt_path, map_location="cpu")
    filtered = {k: v for k, v in state_dict.items() if not ("classifier" in k or "fc" in k)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("Loaded pretrained backbone (features only).")
    if len(unexpected) > 0:
        print("Unexpected keys (ignored):", unexpected[:10])
    if len(missing) > 0:
        print("Missing keys (ok):", missing[:10])



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


def get_probabilities(model, loader, device):
    """
    Sigmoid probabilities
    Output shape: (N, C)
    """
    model.eval()
    probs_all = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            probs_all.append(probs.cpu())

    return torch.cat(probs_all, dim=0)


def evaluate_weighted_ensemble(models_dict, weights, loader, device):
    """
    models_dict: {"name": model}
    weights: {"name": float}
    """
    # Normalizing weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Getting probabilities for each model
    all_probs = {}
    for name, model in models_dict.items():
        all_probs[name] = get_probabilities(model, loader, device)

    # Weighted averaging
    ensemble_probs = torch.zeros_like(next(iter(all_probs.values())))
    for name, probs in all_probs.items():
        ensemble_probs += weights[name] * probs

    # Thresholding
    ensemble_preds = np.zeros_like(ensemble_probs.numpy())
    for c in range(ensemble_probs.shape[1]):
        ensemble_preds[:, c] = (
            ensemble_probs[:, c].numpy() >= THRESHOLDS[c]
        ).astype(int)

    return ensemble_preds

def run_weighted_ensemble_evaluation(test_csv,test_image_dir,batch_size,model_configs,checkpoint_dir="checkpoints",weights=None,
    img_size=256):
    

    print("\n===== Weighted Ensemble Evaluation =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    test_ds = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build and load models
    models_dict = {}
    for name, cfg in model_configs.items():
        
        model = build_model(
            backbone=cfg["backbone"],
            num_classes=3,
            pretrained=True,
            use_mha=cfg.get("use_mha", False),
            use_se=cfg.get("use_se", False),
            num_heads=4
        ).to(device)


        tag = "mha" if cfg.get("use_mha", False) else ("se" if cfg.get("use_se", False) else "base")
        ckpt_path = os.path.join(checkpoint_dir, f"best_{cfg['backbone']}_{tag}.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        models_dict[name] = model

    # Default: equal weights
    if weights is None:
        weights = {name: 1.0 for name in models_dict.keys()}

    # Run ensemble inference
    ensemble_preds = evaluate_weighted_ensemble(
        models_dict=models_dict,
        weights=weights,
        loader=test_loader,
        device=device
    )

    # Ground truth
    y_true = []
    for _, labels in test_loader:
        y_true.append(labels.numpy())
    y_true = np.vstack(y_true)

    # Metrics
    disease_names = ["DR", "Glaucoma", "AMD"]
    results = {}

    for i, disease in enumerate(disease_names):
        results[disease] = {
            "precision": precision_score(y_true[:, i], ensemble_preds[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], ensemble_preds[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], ensemble_preds[:, i], zero_division=0),
            "kappa": cohen_kappa_score(y_true[:, i], ensemble_preds[:, i]),
        }

        print(f"\n{disease} — Weighted Ensemble")
        print("Precision:", results[disease]["precision"])
        print("Recall   :", results[disease]["recall"])
        print("F1       :", results[disease]["f1"])
        print("Kappa    :", results[disease]["kappa"])

    return results



def train_one_backbone(
    backbone,
    train_csv, val_csv, test_csv,
    train_image_dir, val_image_dir, test_image_dir,
    epochs=20, batch_size=32, lr=1e-4, img_size=256,
    save_dir="checkpoints",
    pretrained_backbone=None,
    use_mha=True,
    use_se=False,
    loss_type="bce",   
    phase1_epochs=6,
    phase1_lr=1e-4,
    phase2_lr=1e-5,
    num_heads=4
):  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Training {backbone} | MHA={use_mha} | SE={use_se} | Loss={loss_type}")
    # transforms
    train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, train_transform)
    val_ds   = RetinaMultiLabelDataset(val_csv,   val_image_dir,   eval_transform)
    test_ds  = RetinaMultiLabelDataset(test_csv,  test_image_dir,  eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = build_model(backbone, num_classes=3, pretrained=True, use_mha=use_mha, use_se=use_se, num_heads=num_heads).to(device)

    # checkpoint path
    os.makedirs(save_dir, exist_ok=True)
    tag = "mha" if use_mha else ("se" if use_se else "base")
    ckpt_path = os.path.join(save_dir, f"best_{backbone}_{tag}.pt")

    # load pretrained backbone features (optional)
    if pretrained_backbone is not None:
        load_pretrained_features_only(model, pretrained_backbone)

    # criterion
    # ========================

    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()

    elif loss_type == "class_balanced":
        class_w = compute_class_weights(train_csv).to(device)
        print("Class weights:", class_w.detach().cpu().numpy())
        criterion = ClassBalancedBCELoss(class_w)

    elif loss_type == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)


    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

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

    
    #Testing
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

    #onsite test
    print("\nRunning onsite inference (no labels)...")

    onsite_csv = os.path.join(BASE_DIR, "onsite_test_submission.csv")
    onsite_image_dir = os.path.join(BASE_DIR, "images", "onsite_test")

    onsite_ds = RetinaOnsiteDataset(onsite_csv, onsite_image_dir, eval_transform)
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
    tag = "mha" if use_mha else ("se" if use_se else "base")
    submission_path = os.path.join(BASE_DIR, f"onsite_submission_{backbone}_{tag}.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Onsite submission saved to: {submission_path}")

def run_weighted_ensemble_onsite(
    onsite_csv,
    onsite_image_dir,
    batch_size,
    model_configs,
    checkpoint_dir="checkpoints",
    weights=None,
    img_size=256,
    out_csv="onsite_ensemble_submission.csv"
):
    print("\n===== Weighted Ensemble Onsite Inference =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    onsite_ds = RetinaOnsiteDataset(onsite_csv, onsite_image_dir, transform)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False)

    # Build models
    models_dict = {}
    for name, cfg in model_configs.items():
        model = build_model(
            backbone=cfg["backbone"],
            num_classes=3,
            pretrained=True,
            use_mha=cfg.get("use_mha", False),
            use_se=cfg.get("use_se", False),
            num_heads=4
        ).to(device)

        tag = "mha" if cfg.get("use_mha", False) else ("se" if cfg.get("use_se", False) else "base")
        ckpt_path = os.path.join(checkpoint_dir, f"best_{cfg['backbone']}_{tag}.pt")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models_dict[name] = model

    # Normalize weights
    if weights is None:
        weights = {k: 1.0 for k in models_dict.keys()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Ensemble inference
    submissions = []

    with torch.no_grad():
        for imgs, names in onsite_loader:
            imgs = imgs.to(device)

            ensemble_probs = None
            for name, model in models_dict.items():
                probs = torch.sigmoid(model(imgs))
                if ensemble_probs is None:
                    ensemble_probs = weights[name] * probs
                else:
                    ensemble_probs += weights[name] * probs

            preds = (ensemble_probs.cpu().numpy() > np.array(THRESHOLDS)).astype(int)

            for img_id, p in zip(names, preds):
                submissions.append([img_id, int(p[0]), int(p[1]), int(p[2])])

    submission_df = pd.DataFrame(submissions, columns=["id", "D", "G", "A"])
    submission_df.to_csv(out_csv, index=False)

    print(f"Ensemble submission saved to: {out_csv}")


if __name__ == "__main__":

    train_csv = os.path.join(BASE_DIR, "train.csv")
    val_csv   = os.path.join(BASE_DIR, "val.csv")
    test_csv  = os.path.join(BASE_DIR, "offsite_test.csv")

    train_image_dir = os.path.join(BASE_DIR, "images", "train")
    val_image_dir   = os.path.join(BASE_DIR, "images", "val")
    test_image_dir  = os.path.join(BASE_DIR, "images", "offsite_test")

    pretrained_backbone = os.path.join(
        BASE_DIR, "pretrained_backbone", "ckpt_efficientnet_ep50.pt"
    )

    batch_size = 32

    train_one_backbone(
    backbone="efficientnet",
    train_csv=train_csv,
    val_csv=val_csv,
    test_csv=test_csv,
    train_image_dir=train_image_dir,
    val_image_dir=val_image_dir,
    test_image_dir=test_image_dir,
    epochs=20,
    batch_size=batch_size,
    img_size=256,
    pretrained_backbone=pretrained_backbone,
    use_mha=False,
    use_se=True,         
    loss_type="bce",
    phase1_epochs=7,
    phase1_lr=1e-4,
    phase2_lr=1e-5,
    num_heads=4
)

    train_one_backbone(
    backbone="resnet18",
    train_csv=train_csv,
    val_csv=val_csv,
    test_csv=test_csv,
    train_image_dir=train_image_dir,
    val_image_dir=val_image_dir,
    test_image_dir=test_image_dir,
    epochs=20,
    batch_size=batch_size,
    img_size=256,
    pretrained_backbone=None,
    use_mha=False,         
    use_se=True,           
    loss_type="bce",
    phase1_epochs=7,
    phase1_lr=1e-4,
    phase2_lr=1e-5,
    num_heads=4
)

    model_configs = {
    "efficientnet_se": {
        "backbone": "efficientnet",
        "use_mha": False,
        "use_se": True
    },
    "resnet18_se": {
        "backbone": "resnet18",
        "use_mha": False,
        "use_se": True
    }
}


    weights = {
        "efficientnet_se": 0.55,
        "resnet18_se": 0.45
    }

    run_weighted_ensemble_evaluation(
        test_csv=test_csv,
        test_image_dir=test_image_dir,
        batch_size=batch_size,
        model_configs=model_configs,
        checkpoint_dir="checkpoints",
        weights=weights,
        img_size=256
    )
    run_weighted_ensemble_onsite(
    onsite_csv=os.path.join(BASE_DIR, "onsite_test_submission.csv"),
    onsite_image_dir=os.path.join(BASE_DIR, "images", "onsite_test"),
    batch_size=batch_size,
    model_configs=model_configs,
    checkpoint_dir="checkpoints",
    weights=weights,
    img_size=256,
    out_csv="onsite_ensemble_submission.csv"
    )


