import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


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


# ========================
# build model
# ========================
def build_model(backbone="resnet18", num_classes=3, pretrained=True):

    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model


# ========================
# model training and val
# ========================
def train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, 
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints",pretrained_backbone=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    for p in model.parameters():
        p.requires_grad = True
    
    # loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict)
    

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model for {backbone} at {ckpt_path}")

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
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = torch.tensor(y_true).numpy()
    y_pred = torch.tensor(y_pred).numpy()

    disease_names = ["DR", "Glaucoma", "AMD"]

    for i, disease in enumerate(disease_names):  #compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro",zero_division=0)
        recall = recall_score(y_t, y_p, average="macro",zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro",zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"{disease} Results [{backbone}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")


    
# ========================
# main
# ========================
if __name__ == "__main__":
    train_csv = "train.csv" # replace with your own train label file path
    val_csv   = "val.csv" # replace with your own validation label file path
    test_csv  = "offsite_test.csv"  # replace with your own test label file path
    train_image_dir ="./images/train"   # replace with your own train image floder path
    val_image_dir = "./images/val"  # replace with your own validation image floder path
    test_image_dir = "./images/offsite_test" # replace with your own test image floder path
    pretrained_backbone = './pretrained_backbone/ckpt_resnet18_ep50.pt'  # replace with your own pretrained backbone path
    backbone = 'resnet18'  # backbone choices: ["resnet18", "efficientnet"]
    train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
                           epochs=20, batch_size=32, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)
