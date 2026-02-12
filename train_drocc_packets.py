import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    classification_report, confusion_matrix
)
from tqdm import tqdm

# ============================
# 1) Dataset Class
# ============================
class PacketImageDataset(Dataset):
    def __init__(self, root_dir: str, label: int, img_size: int = 32):
        self.root_dir = root_dir
        self.label = int(label)
        self.img_size = img_size
        if not os.path.exists(root_dir):
            print(f"Warning: Folder not found {root_dir}")
            self.files = []
        else:
            exts = (".png", ".jpg", ".jpeg")
            self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(exts)]
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), 
        ])

    def __len__(self) -> int: return len(self.files)
    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        try:
            img = self.transform(Image.open(img_path))
            return img, self.label
        except:
            return torch.zeros((1, self.img_size, self.img_size)), self.label

# ============================
# 2) Architecture with Bottleneck
# ============================
class LeNet5DROCC(nn.Module):
    def __init__(self, bottleneck_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        
        self.fc1 = nn.Linear(120, 84)
        # Bottleneck forces a compact representation of "Normal"
        self.bottleneck = nn.Linear(84, bottleneck_dim) 
        self.fc2 = nn.Linear(bottleneck_dim, 1)

    def forward(self, x, return_embedding=False):
        x = torch.tanh(self.conv1(x)); x = F.avg_pool2d(x, 2)
        x = torch.tanh(self.conv2(x)); x = F.avg_pool2d(x, 2)
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = torch.tanh(self.fc1(x))
        emb = torch.tanh(self.bottleneck(x))
        
        if return_embedding:
            return emb
        
        return self.fc2(emb)

# ============================
# 3) DROCC Trainer
# ============================
class DROCCTrainer:
    def __init__(self, model, optimizer, lamda, radius, gamma, device):
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.device = device

    def train(self, train_loader, val_loader, total_epochs, only_ce_epochs=5, ascent_step_size=0.001, ascent_num_steps=50):
        best_score = -np.inf
        best_model = None
        
        for epoch in range(total_epochs):
            self.model.train()
            epoch_ce_loss, epoch_adv_loss = 0, 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
            for data, target in pbar:
                data, target = data.to(self.device).float(), target.to(self.device).float()
                self.optimizer.zero_grad()
                
                logits = self.model(data).squeeze()
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                
                loss = ce_loss
                if epoch >= only_ce_epochs:
                    # Anomaly generation only from normal data (label 0)
                    pos_data = data[target == 0]
                    if len(pos_data) > 0:
                        adv_loss = self.one_class_adv_loss(pos_data, ascent_num_steps, ascent_step_size)
                        loss += adv_loss * self.lamda
                        epoch_adv_loss += adv_loss.item()
                
                loss.backward()
                self.optimizer.step()
                epoch_ce_loss += ce_loss.item()
                pbar.set_postfix(CE=f"{ce_loss.item():.4f}", Adv=f"{epoch_adv_loss:.4f}")

            test_auc = self.test(val_loader)
            if test_auc > best_score:
                best_score = test_auc
                best_model = copy.deepcopy(self.model.state_dict())
            print(f" > Epoch {epoch + 1}: AUC: {test_auc:.4f} (Best: {best_score:.4f})")
            
        if best_model:
            self.model.load_state_dict(best_model)

    def one_class_adv_loss(self, x_train, num_steps, step_size):
        # Refined Ascent: Find most 'Normal' looking point outside radius R
        x_adv = (x_train.clone() + torch.randn_like(x_train) * 0.001).detach().requires_grad_(True)
        
        for i in range(num_steps):
            logits = self.model(x_adv).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            with torch.no_grad():
                norm = torch.norm(grad, p=2, dim=(1,2,3), keepdim=True) + 1e-10
                x_adv.add_(step_size * grad / norm)
                
                # Projection to [R, gamma*R]
                h = x_adv - x_train
                h_norm = torch.norm(h, p=2, dim=(1,2,3), keepdim=True) + 1e-10
                proj_scale = torch.clamp(h_norm, self.radius, self.gamma * self.radius) / h_norm
                x_adv.copy_(x_train + proj_scale * h)
                
                # Input clipping for valid image range
                x_adv.clamp_(0, 1)
        
        adv_logits = self.model(x_adv).squeeze()
        return F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        all_scores, all_labels = [], []
        for data, target in loader:
            logits = self.model(data.to(self.device).float()).squeeze()
            all_scores.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(target.numpy())
        return roc_auc_score(all_labels, all_scores)

# ============================
# 4) Main Flow
# ============================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DIR = "pcap_images/Normal"
    TEST_NORM = "pcap_images/test"
    TEST_ATTACK = "pcap_images/Attacks"
    
    # 1. Loading Data
    train_ds = PacketImageDataset(TRAIN_DIR, label=0)
    val_ds = ConcatDataset([PacketImageDataset(TEST_NORM, 0), PacketImageDataset(TEST_ATTACK, 1)])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # 2. Model Prep
    model = LeNet5DROCC(bottleneck_dim=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Radius Improvement: Estimate Radius in Latent Space
    print("Estimating latent radius R...")
    model.eval()
    with torch.no_grad():
        all_embs = []
        # Use a subset or full train loader for estimation
        for i, (data, _) in enumerate(train_loader):
            emb = model(data.to(DEVICE), return_embedding=True)
            all_embs.append(emb)
            if i > 10: break # Use first 10 batches for speed
        
        all_embs = torch.cat(all_embs, dim=0)
        center = torch.mean(all_embs, dim=0)
        distances = torch.norm(all_embs - center, p=2, dim=1)
        # coverage-based radius (e.g., 50% of mean distance)
        estimated_R = torch.mean(distances).item() * 0.5 
    
    print(f"Calculated Latent Radius R: {estimated_R:.4f}")

    # 4. Training
    trainer = DROCCTrainer(model, optimizer, lamda=1.0, radius=estimated_R, gamma=2.0, device=DEVICE)
    trainer.train(train_loader, val_loader, total_epochs=15, only_ce_epochs=5)

    # 5. Evaluation & Visualization
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for d, t in val_loader:
            output = model(d.to(DEVICE).float()).squeeze()
            scores.extend(torch.sigmoid(output).cpu().numpy())
            labels.extend(t.numpy())
    
    scores, labels = np.array(scores), np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    preds = (scores >= best_thresh).astype(int)
    
    print("\n--- Final Report ---")
    print(classification_report(labels, preds, target_names=["Normal", "Anomali"]))
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC: {roc_auc_score(labels, scores):.4f}')
    plt.plot([0,1],[0,1], 'k--')
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(labels, preds)
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()