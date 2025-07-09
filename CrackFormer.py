# crackformer_full.py

import os
import subprocess
import zipfile
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from timm import create_model
import numpy as np
from sklearn.metrics import jaccard_score, f1_score

# ----------------------------
# Step 1: Download datasets from Kaggle
# ----------------------------
def download_datasets():
    datasets = {
        "crack500": "pauldavid22/crack50020220509t090436z001",
        "uav_crack": "ziya07/uav-based-crack-detection-dataset",
        "sc_crack": "lakshaymiddha/crack-segmentation-dataset"
    }
    os.makedirs("datasets", exist_ok=True)
    for name, kaggle_id in datasets.items():
        target_dir = f"datasets/{name}"
        os.makedirs(target_dir, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download", "-d", kaggle_id, "-p", target_dir, "--unzip"
        ])
download_datasets()

# ----------------------------
# Step 2: Crack Dataset Class
# ----------------------------
class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name.replace('.jpg', '_mask.png'))).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)
        return image, mask

# ----------------------------
# Step 3: CrackFormer Architecture
# ----------------------------
class CrackFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rpn = maskrcnn_resnet50_fpn(pretrained=True)
        self.rpn.eval()  # freeze RPN

        self.swin = create_model("swinv2_base_window12_192_22k", pretrained=True, num_classes=0, features_only=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            _ = self.rpn(x)  # simulate RPN region proposals

        features = self.swin.forward_features(x)
        feat = features[-1]
        out = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.decoder(out)

# ----------------------------
# Step 4: Reinforcement Learning Agent
# ----------------------------
class RLAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(512*512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Linear(512*512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        flat = state.view(state.size(0), -1)
        probs = self.policy(flat)
        value = self.value(flat)
        return probs, value

# ----------------------------
# Step 5: IoU and Dice metrics
# ----------------------------
def dice_coef(pred, target):
    pred = (pred > 0.5).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# ----------------------------
# Step 6: Training Pipeline
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrackFormer().to(device)
agent = RLAgent().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
rl_optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

# Load Crack500 dataset
train_dataset = CrackDataset("datasets/crack500/images", "datasets/crack500/masks")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Training loop
for epoch in range(1, 51):
    model.train()
    epoch_loss, epoch_dice, epoch_iou = 0, 0, 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        # RL Agent Decision
        probs, values = agent(outputs.detach())
        action = probs.multinomial(num_samples=1)
        reward = dice_coef(outputs, masks).item()
        returns = torch.tensor([reward], device=device)

        advantage = returns - values.squeeze()
        rl_loss = -torch.log(probs.gather(1, action)).mean() * advantage + F.mse_loss(values.squeeze(), returns)

        # Supervised loss
        loss = criterion(outputs, masks)
        total_loss = loss + 0.1 * rl_loss

        optimizer.zero_grad()
        rl_optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        rl_optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice_coef(outputs, masks).item()
        epoch_iou += iou_score(outputs, masks).item()

    print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}, Dice: {epoch_dice/len(train_loader):.4f}, IoU: {epoch_iou/len(train_loader):.4f}")

# ----------------------------
# Save model
# ----------------------------
torch.save(model.state_dict(), "crackformer_model.pth")
torch.save(agent.state_dict(), "rl_agent.pth")
