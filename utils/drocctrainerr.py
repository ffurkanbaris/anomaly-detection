import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.ensemble import IsolationForest
from tqdm import tqdm


# ============================================================
# 1) Dataset
# ============================================================
class PacketImageDataset(Dataset):
    def __init__(self, root_dir: str, label: int, img_size: int = 32,
                 augment: bool = False):
        self.label    = int(label)
        self.img_size = img_size
        if not os.path.exists(root_dir):
            print(f"[WARN] Directory not found: {root_dir}")
            self.files = []
        else:
            exts = (".png", ".jpg", ".jpeg")
            self.files = sorted(
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith(exts)
            )

        base_tf = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
        if augment:
            base_tf = [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x)),
                transforms.Lambda(lambda x: x.clamp(0, 1)),
            ]
        self.transform = transforms.Compose(base_tf)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx] if self.files else ""
        try:
            img = self.transform(Image.open(path))
            return img, self.label, path
        except Exception:
            return torch.zeros((1, self.img_size, self.img_size)), self.label, path


# ============================================================
# 2) Model
# ============================================================
class MNIST_LeNet(nn.Module):
    """
    LeNet tarzı basit mimari. `img_size`'a göre pool sayısı dinamik:
    uzamsal boyut >= 2 olduğu sürece en fazla 2 kez pool uygulanır.

        28x28 -> Conv -> BN -> LeakyReLU -> Pool (14) -> Conv -> BN -> LeakyReLU -> Pool (7)
        6x6   -> Conv -> BN -> LeakyReLU -> Pool (3)  -> Conv -> BN -> LeakyReLU         (3)
        Flatten -> FC(flat -> 64)  (embedding)
        FC(64 -> 1)                (logit; BCEWithLogits sigmoid uygular)
    """
    def __init__(self, img_size: int = 28):
        super().__init__()

        self.img_size = img_size
        self.rep_dim  = 128

        c1, c2 = 32, 64

        self.pool  = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, c1, 3, bias=False, padding=1)
        self.bn1   = nn.BatchNorm2d(c1, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, bias=False, padding=1)
        self.bn2   = nn.BatchNorm2d(c2, eps=1e-04, affine=False)

        # Uzamsal boyut 2'nin altına düşmeyecek şekilde en fazla 2 pool uygula
        s = img_size
        num_pools = 0
        max_pools = 2
        while num_pools < max_pools and s // 2 >= 2:
            s = s // 2
            num_pools += 1
        self.num_pools = num_pools
        self.feat_side = s
        flat_dim = c2 * s * s

        self.fc1 = nn.Linear(flat_dim, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, 1, bias=False)

    def forward(self, x, return_embedding: bool = False):
        x = x.view(x.shape[0], 1, self.img_size, self.img_size)

        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        if self.num_pools >= 1:
            x = self.pool(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        if self.num_pools >= 2:
            x = self.pool(x)

        x = x.view(x.size(0), -1)
        emb = self.fc1(x)
        return emb if return_embedding else self.fc2(emb)


# ============================================================
# 3) LR Scheduler
# ============================================================
def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
    if epoch < only_ce_epochs:
        lr = learning_rate
    else:
        progress = (epoch - only_ce_epochs) / max(total_epochs - only_ce_epochs, 1)
        lr = learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ============================================================
# 4) DROCCTrainer
# ============================================================
class DROCCTrainer:
    def __init__(self, model, optimizer, lamda, radius, gamma, device):
        self.model     = model
        self.optimizer = optimizer
        self.lamda     = lamda
        self.radius    = radius
        self.gamma     = gamma
        self.device    = device
        self.history   = {
            "train_ce": [],
            "train_adv": [],
            "train_compactness": [],
            "train_mean_dist": [],
            "test_auc": [],
            "test_acc": [],
        }
        self._best_auc        = -float("inf")
        self._best_state_dict = None

    def train(self, train_loader, learning_rate, lr_scheduler, total_epochs,
              only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50,
              test_loader=None):
        
        self.ascent_num_steps  = ascent_num_steps
        self.ascent_step_size  = ascent_step_size

        for epoch in range(total_epochs):
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)

            epoch_adv_loss = 0.0
            epoch_ce_loss  = 0.0
            batch_idx      = -1

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>3}/{total_epochs}", leave=False)

            for data, target, _ in pbar:
                batch_idx += 1
                data   = data.to(self.device).float()
                target = target.to(self.device).float()
                target = torch.squeeze(target)

                self.optimizer.zero_grad()

                logits  = self.model(data)
                logits  = torch.squeeze(logits, dim=1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                epoch_ce_loss += ce_loss.item()

                if epoch >= only_ce_epochs:
                    # Normal örnekleri (label=0) seç, etraflarında adv örnek üret
                    normal_data = data[target == 0]
                    if len(normal_data) > 1:
                        adv_loss = self.one_class_adv_loss(normal_data)
                        epoch_adv_loss += adv_loss.detach().item()
                        loss            = ce_loss + adv_loss * self.lamda
                    else:
                        loss = ce_loss
                else:
                    loss = ce_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                pbar.set_postfix(CE=f"{ce_loss.item():.4f}", Adv=f"{epoch_adv_loss:.4f}")

            n_batches     = batch_idx + 1
            avg_ce        = epoch_ce_loss / n_batches
            avg_adv       = epoch_adv_loss / n_batches
            
            # Eğitim setinin compactness değerini hesapla
            train_proxy   = self._train_proxy(train_loader)

            self.history["train_ce"].append(avg_ce)
            self.history["train_adv"].append(avg_adv)
            self.history["train_compactness"].append(train_proxy)
            
            info = getattr(self, "_last_train_info", {})
            self.history["train_mean_dist"].append(info.get("mean_dist", 0.0))

            mean_d = info.get("mean_dist", 0.0)
            std_d  = info.get("std_dist",  0.0)
            in_r   = info.get("within_r",  0.0)
            
            print(
                f"Epoch {epoch+1:>3}/{total_epochs}  |  "
                f"CE: {avg_ce:.4f}  |  Adv: {avg_adv:.4f}  |  "
                f"Train EmbDist: {mean_d:.4f}±{std_d:.4f}  "
                f"within_R: {in_r*100:.1f}%"
            )

            # İsteğe bağlı: her epoch sonrası test seti üzerinde değerlendirme.
            # Warmup (only_ce_epochs) sırasında test atlanır: model henuz adversarial
            # bilesenden hicbir sinyal almadigi icin AUC olcumu sadece guruluk uretir
            # ve best-model secimini yaniltabilir.
            if test_loader is not None:
                if epoch < only_ce_epochs:
                    self.history["test_auc"].append(float("nan"))
                    self.history["test_acc"].append(float("nan"))
                    print(
                        f"[TEST] Epoch {epoch+1:>3}/{total_epochs}  |  "
                        f"(warmup, skipped — only_ce_epochs={only_ce_epochs})"
                    )
                else:
                    test_scores, test_labels = self.get_scores_and_labels(test_loader)
                    try:
                        test_auc = roc_auc_score(test_labels, test_scores)
                    except ValueError:
                        test_auc = float("nan")
                    test_preds = (test_scores >= 0.5).astype(int)
                    test_acc   = accuracy_score(test_labels, test_preds)

                    self.history["test_auc"].append(test_auc)
                    self.history["test_acc"].append(test_acc)

                    print(
                        f"[TEST] Epoch {epoch+1:>3}/{total_epochs}  |  "
                        f"AUC: {test_auc:.4f}  |  Acc: {test_acc:.4f}"
                    )

                    # En iyi modeli AUC'ye göre sakla
                    if np.isfinite(test_auc) and test_auc > self._best_auc:
                        self._best_auc        = test_auc
                        self._best_state_dict = copy.deepcopy(self.model.state_dict())

        # Eğitim sonunda en iyi modeli geri yükle (varsa)
        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)
            print(f"\n[INFO] Best model restored (AUC={self._best_auc:.4f}).")

        print("\n[DONE] Training complete.")

    def one_class_adv_loss(self, x_train_data):
        """Normal=0, Attack=1 konvansiyonunda adversarial kayıp.

        - Normal örnek etrafına küçük gürültü ekleyip gradient ascent ile
          sınır bölgesindeki zor örnekler bulunur.
        - Hedef sınıf = attack (1): BCE(logits, 1)'i maksimize etmek,
          modelin çıkışını 0'a (normal'e) yaklaştıracak yönde perturbasyon
          üretir — yani "normale en çok benzeyen saldırı örnekleri".
        - Son adımda bu zor örnekler attack (1) olarak etiketlenip model
          eğitilir → karar sınırı normal manifoldunun etrafında sıkılaşır.
        """
        batch_size = len(x_train_data)

        # Başlangıç tensörünü temiz bir şekilde gradyan takibine al
        x_adv_sampled = (x_train_data + torch.randn_like(x_train_data)).detach().requires_grad_(True)

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():
                # Adv örnekler için hedef = attack (1)
                new_targets = torch.ones(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets).float()

                logits    = self.model(x_adv_sampled)
                logits    = torch.squeeze(logits, dim=1)
                new_loss  = F.binary_cross_entropy_with_logits(logits, new_targets)

                # Türev hesaplama işlemi
                grad      = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_normalized = grad / (grad_norm + 1e-10)

            # Tensörün piksellerini gradient yönünde güncelliyoruz
            with torch.no_grad():
                x_adv_sampled = x_adv_sampled + self.ascent_step_size * grad_normalized
                x_adv_sampled = x_adv_sampled.clamp(0, 1)
            
            # 2. KRİTİK DÜZELTME: Güncellenen tensörü tekrar gradyan takibine sokuyoruz!
            x_adv_sampled = x_adv_sampled.detach().requires_grad_(True)

            # Her 10 adımda bir hipersfer yüzeyine (Radius sınırına) yansıtma (Projection)
            if (step + 1) % 10 == 0:
                h      = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h ** 2, dim=tuple(range(1, h.dim()))))
                alpha  = torch.clamp(norm_h, self.radius, self.gamma * self.radius).to(self.device)
                proj          = (alpha / (norm_h + 1e-10)).view(-1, *[1] * (h.dim() - 1))
                h             = proj * h
                x_adv_sampled = (x_train_data + h).clamp(0, 1).detach().requires_grad_(True)

        # Adv örnekler attack sınıfı (1) olarak eğitilir
        adv_targets = torch.ones(batch_size, 1).to(self.device)
        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, torch.squeeze(adv_targets))
        return adv_loss

    @torch.no_grad()
    def _train_proxy(self, loader) -> float:
        self.model.eval()
        all_embs = []
        for data, _, _ in loader:
            emb = self.model(data.to(self.device).float(), return_embedding=True)
            all_embs.append(emb.cpu())

        if not all_embs: return 0.0

        all_embs  = torch.cat(all_embs, dim=0)
        centroid  = all_embs.mean(dim=0)
        dists     = torch.norm(all_embs - centroid, p=2, dim=1)

        mean_dist = dists.mean().item()
        std_dist  = dists.std().item()
        within_r  = (dists <= self.radius).float().mean().item()

        self._last_train_info = {
            "mean_dist" : mean_dist,
            "std_dist"  : std_dist,
            "within_r"  : within_r,
        }
        return -mean_dist

    @torch.no_grad()
    def get_scores_and_labels(self, loader):
        self.model.eval()
        scores, labels = [], []
        for data, target, _ in loader:
            logits = self.model(data.to(self.device).float())
            logits = torch.squeeze(logits, dim=1)
            scores.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(target.numpy())
        return np.array(scores), np.array(labels)

    @torch.no_grad()
    def get_embeddings(self, loader):
        self.model.eval()
        embs, labels = [], []
        for data, target, _ in loader:
            e = self.model(data.to(self.device).float(), return_embedding=True)
            embs.extend(e.cpu().numpy())
            labels.extend(target.numpy())
        return np.array(embs), np.array(labels)


# ============================================================
# 5) Evaluation & Plotting
# ============================================================
def evaluate_and_plot(trainer: DROCCTrainer, test_loader: DataLoader, save_path: str = "final_model_performance1.png"):
    print("\n" + "=" * 60)
    print("  EVALUATION (TEST SET)")
    print("=" * 60)

    # Threshold selection directly on test set
    test_scores, test_labels = trainer.get_scores_and_labels(test_loader)
    fpr, tpr, thresh = roc_curve(test_labels, test_scores)
    
    best_idx    = np.argmax(tpr - fpr)
    best_idx    = min(best_idx, len(thresh) - 1) if len(thresh) > 0 else 0
    best_thresh = float(thresh[best_idx]) if len(thresh) > 0 else 0.5
    
    print(f"[TEST] Optimal threshold (Youden J): {best_thresh:.6f}")

    preds = (test_scores >= best_thresh).astype(int)

    auc  = roc_auc_score(test_labels, test_scores)
    acc  = accuracy_score(test_labels, preds)
    f1   = f1_score(test_labels, preds, zero_division=0)
    prec = precision_score(test_labels, preds, zero_division=0)
    rec  = recall_score(test_labels, preds, zero_division=0)

    print(classification_report(test_labels, preds, target_names=["Normal", "Attack"]))

    # Isolation Forest baseline
    print("[BASELINE] Training Isolation Forest on embeddings ...")
    train_embs, _ = trainer.get_embeddings(
        DataLoader(trainer._train_ds_ref, batch_size=256,
                   collate_fn=lambda b: (
                       torch.stack([x[0] for x in b]),
                       torch.tensor([x[1] for x in b]),
                       [x[2] for x in b]
                   ))
    )
    test_embs, _  = trainer.get_embeddings(test_loader)
    iso       = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iso.fit(train_embs)
    iso_scores = -iso.score_samples(test_embs)
    iso_auc    = roc_auc_score(test_labels, iso_scores)
    print(f"[BASELINE] Isolation Forest AUC: {iso_auc:.4f}")

    # Figure
    fig = plt.figure(figsize=(24, 10))
    fig.suptitle("DROCC Anomaly Detector – Final Evaluation (Train & Test Only)", fontsize=15, weight="bold")

    # 1. ROC Curve
    ax1 = fig.add_subplot(1, 5, 1)
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"DROCC AUC={auc:.4f}")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    fpr_iso, tpr_iso, _ = roc_curve(test_labels, iso_scores)
    ax1.plot(fpr_iso, tpr_iso, color="green", lw=1.5, linestyle="--", label=f"IsoForest AUC={iso_auc:.4f}")
    ax1.scatter(fpr[best_idx], tpr[best_idx], color="red", s=80, zorder=5, label="Optimal threshold")
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve (Test Set)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. Confusion Matrix
    ax2 = fig.add_subplot(1, 5, 2)
    cm = confusion_matrix(test_labels, preds)
    ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    thresh_c = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax2.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                 color="white" if cm[i, j] > thresh_c else "black", fontsize=12)
    ax2.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"], xlabel="Predicted", ylabel="True",
            title=f"Confusion Matrix\n(thresh={best_thresh:.3f})")

    # 3. Metrics summary
    ax3 = fig.add_subplot(1, 5, 3)
    ax3.axis("off")
    summary = (
        "── MODEL PERFORMANCE (TEST) ──\n\n"
        f"{'AUC':<14} {auc:.4f}\n"
        f"{'Accuracy':<14} {acc:.4f}\n"
        f"{'F1-Score':<14} {f1:.4f}\n"
        f"{'Precision':<14} {prec:.4f}\n"
        f"{'Recall':<14} {rec:.4f}\n\n"
        "── BASELINE (IsoForest) ──\n\n"
        f"{'AUC':<14} {iso_auc:.4f}\n\n"
        "── Threshold ──\n\n"
        f"{'Value':<14} {best_thresh:.6f}\n"
        f"{'TPR':<14} {tpr[best_idx]:.4f}\n"
        f"{'FPR':<14} {fpr[best_idx]:.4f}\n"
    )
    ax3.text(0.05, 0.95, summary, fontsize=11, family="monospace", va="top",
             transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    # 4. Score Distribution
    ax4 = fig.add_subplot(1, 5, 4)
    bins = np.linspace(0, 1, 40)
    ax4.hist(test_scores[test_labels == 0], bins=bins, alpha=0.6, color="steelblue", label="Normal")
    ax4.hist(test_scores[test_labels == 1], bins=bins, alpha=0.6, color="tomato", label="Attack")
    ax4.axvline(best_thresh, color="black", linestyle="--", label=f"Threshold={best_thresh:.3f}")
    ax4.set(xlabel="Anomaly Score", ylabel="Count", title="Score Distribution (Test Set)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # 5. Training History
    ax5 = fig.add_subplot(1, 5, 5)
    epochs = range(1, len(trainer.history["train_compactness"]) + 1)
    ax5.plot(epochs, trainer.history["train_mean_dist"], "o-", color="darkorange", label="Train Emb mean dist (↓)")
    ax5_r = ax5.twinx()
    ax5_r.plot(epochs, trainer.history["train_ce"], "s--", color="steelblue", label="Train CE")
    ax5.set(xlabel="Epoch", ylabel="Train Emb Mean Dist", title="Training History")
    ax5.yaxis.label.set_color("darkorange")
    ax5_r.set_ylabel("Train CE Loss", color="steelblue")
    lines1, labs1 = ax5.get_legend_handles_labels()
    lines2, labs2 = ax5_r.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower right")
    ax5.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[SAVED] Figure → {save_path}")
    plt.show()


# ============================================================
# 6) Main
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DROCC Trainer (Train & Test Only)")
    parser.add_argument("--train_normal_dir", default="cicpcapimages/train")
    parser.add_argument("--test_normal_dir",  default="cicpcapimages/test/normal")
    parser.add_argument("--test_attack_dir",  default="cicpcapimages/test/attack")
    parser.add_argument("--img_size",         type=int,   default=6)
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--epochs",           type=int,   default=10)
    parser.add_argument("--only_ce_epochs",   type=int,   default=2)
    parser.add_argument("--lr",               type=float, default=0.0001)
    parser.add_argument("--lamda",            type=float, default=0.2)
    parser.add_argument("--radius",           type=float, default=1.0)
    parser.add_argument("--gamma",            type=float, default=2.0)
    parser.add_argument("--ascent_step_size", type=float, default=0.001)
    parser.add_argument("--ascent_num_steps", type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=123)
    parser.add_argument("--max_train",        type=int,   default=0,
                        help="Egitim seti icin ust sinir (0 = sinirsiz).")
    parser.add_argument("--max_test",         type=int,   default=10000,
                        help="Test seti (normal+attack birlesik) icin ust sinir (0 = sinirsiz).")
    parser.add_argument("--output_fig",       default="final_model_performance.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {DEVICE}")

    def _cap_dataset(ds, max_n, seed):
        """Rastgele (seed'li) indeks secerek dataset'i max_n ile sinirla."""
        if max_n is None or max_n <= 0 or len(ds) <= max_n:
            return ds
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ds))[:max_n].tolist()
        return Subset(ds, idx)

    # ── Datasets ─────────────────────────────────────────────
    train_ds_full = PacketImageDataset(
        args.train_normal_dir, label=0, img_size=args.img_size, augment=True,
    )
    test_normal_full = PacketImageDataset(
        args.test_normal_dir, label=0, img_size=args.img_size,
    )
    test_attack_full = PacketImageDataset(
        args.test_attack_dir, label=1, img_size=args.img_size,
    )

    # Test icin sinif dengesini koru: max_test/2 normal + max_test/2 attack
    half_test = args.max_test // 2 if args.max_test and args.max_test > 0 else 0
    test_normal = _cap_dataset(test_normal_full, half_test, args.seed)
    test_attack = _cap_dataset(test_attack_full, half_test, args.seed + 1)
    train_ds    = _cap_dataset(train_ds_full,    args.max_train, args.seed + 2)

    test_ds = ConcatDataset([test_normal, test_attack])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Train(normal): {len(train_ds)}/{len(train_ds_full)} | "
          f"Test(normal): {len(test_normal)}/{len(test_normal_full)} | "
          f"Test(attack): {len(test_attack)}/{len(test_attack_full)}")

    # ── Model & Optimizer ────────────────────────────────────
    model = MNIST_LeNet(img_size=args.img_size).to(DEVICE)
    print(f"[INFO] Arch: MNIST_LeNet | img={args.img_size} | rep_dim={model.rep_dim} | "
          f"num_pools={model.num_pools} | feat_side={model.feat_side} | "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ── Trainer ──────────────────────────────────────────────
    trainer = DROCCTrainer(
        model     = model,
        optimizer = optimizer,
        lamda     = args.lamda,
        radius    = args.radius,
        gamma     = args.gamma,
        device    = DEVICE,
    )
    trainer._train_ds_ref = train_ds

    # ── Train ────────────────────────────────────────────────
    trainer.train(
        train_loader      = train_loader,
        learning_rate     = args.lr,
        lr_scheduler      = adjust_learning_rate,
        total_epochs      = args.epochs,
        only_ce_epochs    = args.only_ce_epochs,
        ascent_step_size  = args.ascent_step_size,
        ascent_num_steps  = args.ascent_num_steps,
        test_loader       = test_loader,
    )

    # ── Evaluate ─────────────────────────────────────────────
    evaluate_and_plot(trainer, test_loader, save_path=args.output_fig)

    # ── Save ─────────────────────────────────────────────────
    torch.save(model.state_dict(), "drocc_final_model.pth")
    print("[SAVED] Model weights → drocc_final_model.pth")