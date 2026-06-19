import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import argparse
import os
import glob
from sklearn.metrics import roc_auc_score, roc_curve

def _flat_dim(img_size: int) -> int:
    h = img_size // 2
    return 32 * h * h

class SVDDEncoder(nn.Module):
    def __init__(self, img_size: int = 6, rep_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.rep_dim  = rep_dim
        flat          = _flat_dim(img_size)

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16, eps=1e-4, affine=False)
        self.pool  = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32, eps=1e-4, affine=False)

        self.fc1 = nn.Linear(flat, rep_dim, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), 1, self.img_size, self.img_size)
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

class SVDDDecoder(nn.Module):
    def __init__(self, img_size: int = 6, rep_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        h             = img_size // 2
        self.h        = h
        flat          = 32 * h * h

        self.fc_d  = nn.Linear(rep_dim, flat, bias=False)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1, bias=False)
        self.bn_d1   = nn.BatchNorm2d(16, eps=1e-4, affine=False)
        op = img_size - h * 2
        self.deconv2 = nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0,
                                           output_padding=op, bias=False)

    def forward(self, z):
        x = self.fc_d(z)
        x = x.view(x.size(0), 32, self.h, self.h)
        x = F.leaky_relu(self.bn_d1(self.deconv1(x)))
        return torch.sigmoid(self.deconv2(x))

class SVDDAutoencoder(nn.Module):
    def __init__(self, img_size: int = 6, rep_dim: int = 128):
        super().__init__()
        self.encoder = SVDDEncoder(img_size, rep_dim)
        self.decoder = SVDDDecoder(img_size, rep_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class CustomImageDataset(Dataset):
    def __init__(self, normal_dir=None, attack_dir=None, transform=None):
        self.transform = transform
        self.filepaths = []
        self.labels    = []

        if normal_dir and os.path.exists(normal_dir):
            files = glob.glob(os.path.join(normal_dir, "*.png"))
            self.filepaths.extend(files)
            self.labels.extend([1] * len(files))

        if attack_dir and os.path.exists(attack_dir):
            files = glob.glob(os.path.join(attack_dir, "*.png"))
            self.filepaths.extend(files)
            self.labels.extend([0] * len(files))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img   = Image.open(self.filepaths[idx]).convert("L")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

@torch.no_grad()
def init_center(encoder, loader, device, eps=0.1):
    encoder.eval()
    all_z = []
    for imgs, labels in loader:
        imgs  = imgs.to(device, dtype=torch.float32)
        mask  = (labels == 1)
        imgs  = imgs[mask]
        if imgs.size(0) == 0:
            continue
        z = encoder(imgs)
        all_z.append(z)
    c = torch.cat(all_z, dim=0).mean(dim=0)
    
    c[(torch.abs(c) < eps) & (c < 0)] = -eps
    c[(torch.abs(c) < eps) & (c >= 0)] = eps
    return c

def pretrain(ae, loader, optimizer, device, epochs):
    ae.train()
    print(f"\n[Ön-Eğitim] Autoencoder {epochs} epoch eğitiliyor...")
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device, dtype=torch.float32)
            mask = (labels == 1)
            imgs = imgs[mask]
            if imgs.size(0) == 0:
                continue
            optimizer.zero_grad()
            recon = ae(imgs)
            
            recon_errors = torch.sum((recon - imgs) ** 2, dim=tuple(range(1, recon.dim())))
            loss = torch.mean(recon_errors)
            
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)
            n     += imgs.size(0)
        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"  Ön-Eğitim Epoch [{ep:3d}/{epochs}]  Recon Loss: {total/max(n,1):.5f}")
    print("[Ön-Eğitim] Tamamlandı.\n")

def train_svdd_epoch(encoder, center, loader, optimizer, device):
    encoder.train()
    total, n = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, dtype=torch.float32)
        mask = (labels == 1)
        imgs = imgs[mask]
        if imgs.size(0) == 0:
            continue
        optimizer.zero_grad()
        z    = encoder(imgs)
        dist = torch.sum((z - center) ** 2, dim=1)
        loss = torch.mean(dist)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
        n     += imgs.size(0)
    return total / max(n, 1)

@torch.no_grad()
def evaluate(encoder, center, loader, device):
    encoder.eval()
    all_scores, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, dtype=torch.float32)
        z    = encoder(imgs)
        dist = torch.sum((z - center) ** 2, dim=1)
        score = -dist.cpu().numpy()
        all_scores.append(score)
        all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)

def compute_metrics(scores, labels):
    auc = roc_auc_score(labels, scores)

    print("\n" + "=" * 50)
    print("  [Deep SVDD] SONUÇLAR (Makale Protokolü)")
    print("=" * 50)
    print(f"   ➤ ROC-AUC Skoru : {auc:.4f} (1.0 = Kusursuz, 0.5 = Rastgele)")
    print("=" * 50 + "\n")

    return dict(auc=auc)

def plot_training(history, save_dir):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["loss"], color="steelblue", linewidth=2, label="SVDD Loss")
    axes[0].set_title("Deep SVDD – Eğitim Kaybı")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, history["val_auc"], color="darkorange", linewidth=2, label="Val ROC-AUC")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Deep SVDD – Doğrulama ROC-AUC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(save_dir, "svdd_training_history.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[Başarılı] Eğitim grafikleri kaydedildi: {path}")

def plot_evaluation(scores, labels, save_dir):
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    auc = roc_auc_score(labels, scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pos_scores, bins=50, alpha=0.6, color="green", density=True, label="Normal (1)")
    axes[0].hist(neg_scores, bins=50, alpha=0.6, color="red",   density=True, label="Attack (0)")
    axes[0].set_title("Deep SVDD – Skor Dağılımı")
    axes[0].set_xlabel("Normallik Skoru (Yüksek = Normal)")
    axes[0].set_ylabel("Yoğunluk")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate (FPR)")
    axes[1].set_ylabel("True Positive Rate (TPR)")
    axes[1].set_title("Receiver Operating Characteristic (ROC)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(save_dir, "svdd_evaluation_results.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[Başarılı] Değerlendirme grafikleri (Dağılım & ROC) kaydedildi: {path}")

def get_data_dirs(args):
    if args.img_size == 7:
        root = os.path.join(args.data_path, "network_traffic_7x7_images")
    else:
        root = os.path.join(args.data_path, "wustlehms_images_onehot")

    return (
        os.path.join(root, "train", "normal"),
        os.path.join(root, "test",  "normal"),
        os.path.join(root, "test",  "attack"),
    )

def main(args, device):
    sz = args.img_size

    transform = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
    ])

    train_normal_dir, test_normal_dir, test_attack_dir = get_data_dirs(args)

    train_dataset = CustomImageDataset(normal_dir=train_normal_dir, transform=transform)
    test_dataset  = CustomImageDataset(normal_dir=test_normal_dir,
                                       attack_dir=test_attack_dir, transform=transform)

    n_normal = sum(1 for l in test_dataset.labels if l == 1)
    n_attack = sum(1 for l in test_dataset.labels if l == 0)
    print(f"[Veri] Görüntü boyutu : {sz}x{sz}")
    print(f"[Veri] Train normal   : {len(train_dataset)}")
    print(f"[Veri] Test  normal   : {n_normal}  |  attack: {n_attack}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    encoder    = SVDDEncoder(img_size=sz, rep_dim=args.rep_dim).to(device)
    center_path = os.path.join(args.model_dir, "svdd_center.pt")
    model_path  = os.path.join(args.model_dir, "svdd_encoder.pt")

    if args.eval == 1:
        if not os.path.exists(model_path) or not os.path.exists(center_path):
            print("[Hata] Kayıtlı model veya merkez bulunamadı.")
            return
        encoder.load_state_dict(torch.load(model_path, map_location=device))
        center = torch.load(center_path, map_location=device)
        print(f"[Bilgi] Model ve merkez yüklendi.")
        scores, labels = evaluate(encoder, center, test_loader, device)
        compute_metrics(scores, labels)
        plot_evaluation(scores, labels, args.model_dir)
        return

    if args.pretrain_epochs > 0:
        ae      = SVDDAutoencoder(img_size=sz, rep_dim=args.rep_dim).to(device)
        opt_ae  = optim.Adam(ae.parameters(), lr=args.lr, weight_decay=1e-6)
        pretrain(ae, train_loader, opt_ae, device, args.pretrain_epochs)
        encoder.load_state_dict(ae.encoder.state_dict())
        del ae

    print("[Bilgi] Hiperküre merkezi hesaplanıyor...")
    center = init_center(encoder, train_loader, device)
    print(f"[Bilgi] Merkez norm: {center.norm().item():.4f}")

    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history  = {"loss": [], "val_auc": []}
    best_auc = 0.0

    print(f"\n{'='*50}")
    print(f"  Deep SVDD EĞİTİMİ — {sz}x{sz} görüntüler")
    print(f"  Epoch: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size} | Rep: {args.rep_dim}")
    print(f"{'='*50}\n")

    for epoch in range(1, args.epochs + 1):
        loss = train_svdd_epoch(encoder, center, train_loader, optimizer, device)
        scheduler.step()

        scores, labels = evaluate(encoder, center, test_loader, device)
        auc = roc_auc_score(labels, scores)

        history["loss"].append(loss)
        history["val_auc"].append(auc)

        print(f"Epoch [{epoch:3d}/{args.epochs}]  SVDD Loss: {loss:.6f}  Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(encoder.state_dict(), model_path)
            torch.save(center, center_path)

    print(f"\n[Başarılı] En iyi AUC: {best_auc:.4f}")
    print(f"[Başarılı] Model: {model_path}  |  Merkez: {center_path}")

    plot_training(history, args.model_dir)

    encoder.load_state_dict(torch.load(model_path, map_location=device))
    center = torch.load(center_path, map_location=device)
    scores, labels = evaluate(encoder, center, test_loader, device)
    compute_metrics(scores, labels)
    plot_evaluation(scores, labels, args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep SVDD – DROCC-LF ile kıyaslama için one-class anomali tespiti"
    )
    parser.add_argument("--img_size",       type=int,   default=6,
                        help="Görüntü boyutu: 6 (wustlehms) veya 7 (network_traffic)")
    parser.add_argument("--rep_dim",        type=int,   default=128,
                        help="Encoder çıktı boyutu (DROCC rep_dim ile aynı)")
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--epochs",         type=int,   default=10,
                        help="SVDD eğitim epoch sayısı")
    parser.add_argument("--pretrain_epochs",type=int,   default=10,
                        help="Autoencoder ön-eğitim epoch sayısı (0 = devre dışı)")
    parser.add_argument("--lr",             type=float, default=0.001)
    parser.add_argument("--model_dir",      type=str,   default="svdd_log")
    parser.add_argument("--eval",           type=int,   default=0,
                        help="0=Eğit, 1=Kayıtlı modeli yükle ve değerlendir")
    parser.add_argument("--data_path",      type=str,   default=".",
                        help="Veri dizininin kök yolu")

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.model_dir, exist_ok=True)
    print(f"[Bilgi] Cihaz: {device}")
    torch.set_printoptions(precision=5)

    main(args, device)
