from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import h5py
import numpy as np
import random


# ===== Reproducibility ====
def set_seed(seed = 234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==== Dataset Class ====
class MiniPCamDataset(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path, transform = None):
        with h5py.File(x_path, 'r') as f_x:
            self.images = f_x['x'][:]
        with h5py.File(y_path, 'r') as f_y:
            self.labels = f_y['y'][:].flatten()
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype = torch.float32)


# ==== Model Architecture ====
class PCamResNet(nn.Module):
    def __init__(self):
        super(PCamResNet, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
    

# ==== Training Function ====
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs, device, patience = 5):
    best_recall = 0.0
    epochs_no_improve = 0
    checkpoint_path = './models/best_model.pth'

    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # validation phase
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        v_rec = recall_score(val_targets, val_preds, zero_division=0)

        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Recall: {v_rec:.4f}")

        # early stopping based on recall
        if v_rec > best_recall:
            best_recall = v_rec
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
            print(f"--> Recall improved. Saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    return best_recall

# ==== Main Logic ====
def main():
    set_seed(234)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # define transformations
    ## imagenet standard normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load data
    train_loader = DataLoader(MiniPCamDataset('./data/small_train_x.h5', './data/small_train_y.h5', train_transforms), batch_size=64, shuffle = True)
    val_loader = DataLoader(MiniPCamDataset('./data/small_validation_x.h5', './data/small_validation_y.h5', val_transforms), batch_size=64, shuffle = False)

    model = PCamResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Train Head (Backbone frozen)
    print("\n ==== Training Head ====")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.head.parameters(), lr = 0.0005)

    train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs = 5, device = device, patience = 5)
    
    # Fine Tuning (Unfreeze layer4)
    print("\n ==== Fine Tuning Layer 4 ====")
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs = 20, device = device, patience = 5)

if __name__ == '__main__':
    main()