import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

THRESHOLDS = [0.40, 0.30, 0.45]

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

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(csv_file):
    """
    Here I am computing class weights for multi-label BCE loss w_c = N / (C * n_c)
    """
    df = pd.read_csv(csv_file)
    labels = df.iloc[:, 1:].values  

    N = labels.shape[0]
    C = labels.shape[1]

    class_counts = labels.sum(axis=0)
    class_weights = N / (C * class_counts)

    return torch.tensor(class_weights, dtype=torch.float32)

class ClassBalancedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super(ClassBalancedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(
            weight=class_weights,
            reduction="mean"
        )

    def forward(self, logits, targets):
        return self.bce(logits, targets)
    



class SpatialMHA(nn.Module):
    """
    Multi-Head Self-Attention over spatial feature maps
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


# ========================
# build model
# ========================

def build_model(backbone="resnet18", num_classes=3, pretrained=True, use_mha=False):

    if backbone == "resnet18":
        if use_mha:
           model = ResNet18WithMHA(num_classes=num_classes,pretrained=pretrained,num_heads=4)
        else:
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "efficientnet":
        if use_mha:
           model = EfficientNetWithMHA(num_classes=num_classes,pretrained=pretrained,num_heads=4)
        else:
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError("Unsupported backbone")

    return model



# ========================
# model training and val
# ========================
def train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, 
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints",pretrained_backbone=None,
                       mode="full_finetune", loss_type="bce", focal_gamma=1.0,focal_alpha=1.0,
                       ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # transforms
    transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
    ])


    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = build_model(backbone, num_classes=3, pretrained=True,use_mha=True).to(device)

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained backbone")
    

    # loss & optimizer

    if mode == "freeze_backbone":
        print("Mode: FROZEN BACKBONE (classifier + attention only)")
        for name, p in model.named_parameters():
            if backbone == "resnet18":
                if not (name.startswith("fc") or name.startswith("attn")):
                  p.requires_grad = False
            elif backbone == "efficientnet":
                 if not (name.startswith("classifier") or name.startswith("attn")):
                    p.requires_grad = False


    elif mode == "full_finetune":
        print("Mode: FULL FINE-TUNING (backbone + classifier)")
        for p in model.parameters():
            p.requires_grad = True

    # loss function
    if loss_type == "focal":
        print(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
        criterion = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean"
        )

    elif loss_type == "class_balanced":
        print("Using Class-Balanced BCE Loss")

        class_weights = compute_class_weights(train_csv).to(device)
        print("Class Weights:", class_weights.cpu().numpy())

        criterion = ClassBalancedBCELoss(class_weights)

    else:
        print("Using Standard BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)


    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")


    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                loss = criterion(model(imgs), labels)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch+1}/{epochs} "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print("Saved best model")

    # ========================
    # testing
    # ========================
    
     
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > THRESHOLDS).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = torch.tensor(y_true).numpy()
    y_pred = torch.tensor(y_pred).numpy()

    disease_names = ["DR", "Glaucoma", "AMD"]

    for i, disease in enumerate(disease_names):  #compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p,zero_division=0)
        recall = recall_score(y_t, y_p,zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"{disease} Results [{backbone}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")

    # ========================
    # ONSITE TEST INFERENCE (NO LABELS)
    # ========================
    print("\nRunning onsite inference (no labels)...")

    onsite_csv = os.path.join(BASE_DIR, "onsite_test_submission.csv")
    onsite_image_dir = os.path.join(BASE_DIR, "images", "onsite_test")

    onsite_ds = RetinaOnsiteDataset(onsite_csv, onsite_image_dir, transform)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size,shuffle=False,num_workers=0)

    onsite_predictions = []

    with torch.no_grad():
        for imgs, img_names in onsite_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > THRESHOLDS).astype(int)

            for name, p in zip(img_names, preds):
                onsite_predictions.append([name, p[0], p[1], p[2]])
    submission = pd.DataFrame(
        onsite_predictions,
        columns=["id", "D", "G", "A"]
    )

    submission_path = os.path.join(BASE_DIR, "onsite_submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Onsite submission saved to: {submission_path}")

   
    
# ========================
# main
# ========================
if __name__ == "__main__":
    train_csv = os.path.join(BASE_DIR, "train.csv") # replace with your own train label file path
    val_csv   = os.path.join(BASE_DIR, "val.csv") # replace with your own validation label file path
    test_csv  = os.path.join(BASE_DIR, "offsite_test.csv")  # replace with your own test label file path
    train_image_dir = os.path.join(BASE_DIR, "images", "train")  # replace with your own train image floder path
    val_image_dir   = os.path.join(BASE_DIR, "images", "val")  # replace with your own validation image floder path
    test_image_dir  = os.path.join(BASE_DIR, "images", "offsite_test") # replace with your own test image floder path

    pretrained_backbone = os.path.join(BASE_DIR, "pretrained_backbone", "ckpt_resnet18_ep50.pt" )# replace with your own pretrained backbone path
    backbone = 'resnet18'  # backbone choices: ["resnet18", "efficientnet"]
    train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
                           epochs=20, batch_size=24, lr=1e-4, img_size=256, pretrained_backbone=None,
                           mode="full_finetune",loss_type="bce",focal_gamma=1.0)
