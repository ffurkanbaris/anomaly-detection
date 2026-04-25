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
    classification_report, confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

# ============================================================
# 0) Orijinal DROCC-LF Matematiksel Fonksiyonları
# ============================================================
def normalize_grads(grad):
    grad_norm = torch.sum(torch.abs(grad), dim=1)
    grad_norm = torch.unsqueeze(grad_norm, dim=1)
    grad_norm = grad_norm.repeat(1, grad.shape[1])
    # Sayısal kararlılık için küçük bir epsilon ekle
    grad = grad / (grad_norm + 1e-10) * grad.shape[1]
    return grad

def compute_mahalanobis_distance(grad, diff, radius, device, gamma):
    mhlnbs_dis = torch.sqrt(torch.sum(grad*diff**2, dim=1))
    lamda = torch.zeros((grad.shape[0],1))
    lamda[mhlnbs_dis < radius] = 1
    lamda[mhlnbs_dis > (gamma * radius)] = 2
    return lamda, mhlnbs_dis

def check_left_part1(lam, grad, diff, radius, device):
    n1 = diff**2 * lam**2 * grad**2
    d1 = (1 + lam * grad)**2 + 1e-10
    term = n1/d1
    return torch.sum(term)

def check_left_part2(nu, grad, diff, radius, device, gamma):
    n1 = diff**2 * grad**2
    d1 = (nu + grad)**2 + 1e-10
    term = n1/d1
    return torch.sum(term)

def check_right_part1(lam, grad, diff, radius, device):
    n1 = grad
    d1 = (1 + lam * grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum > radius**2:
        return check_left_part1(lam, grad, diff, radius, device)
    else:
        return np.inf

def check_right_part2(nu, grad, diff, radius, device, gamma):
    n1 = grad*nu**2
    d1 = (nu + grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum < (gamma*radius)**2:
        return check_left_part2(nu, grad, diff, radius, device, gamma)
    else:
        return np.inf

def range_lamda_lower(grad):
    lam, _ = torch.max(grad, dim=1)
    eps, _ = torch.min(grad, dim=1)
    # Sıfıra çok yakın değerlerde taşma/inf oluşmasını engelle
    lam = -1 / (lam + 1e-10) + eps * 0.0001
    return lam

def range_nu_upper(grad, mhlnbs_dis, radius, gamma):
    alpha = (gamma * radius) / (mhlnbs_dis + 1e-10)
    # alpha'yı [0, 0.99] aralığında kırp, 1'e çok yaklaşınca nu patlamasın
    alpha = torch.clamp(alpha, min=0.0, max=0.99)
    max_sigma, _ = torch.max(grad, dim=1)
    nu = (alpha / (1 - alpha)) * max_sigma
    return nu

def optim_solver(grad, diff, radius, device, gamma=2):
    """
    Solver for the optimization problem presented in Proposition 1 in
    https://arxiv.org/abs/2002.12718
    """
    lamda, mhlnbs_dis = compute_mahalanobis_distance(grad, diff, radius, device, gamma)
    lamda_lower_limit = range_lamda_lower(grad).detach().cpu().numpy()
    nu_upper_limit = range_nu_upper(grad, mhlnbs_dis, radius, gamma).detach().cpu().numpy()
    
    #num of values of lamda and nu samples in the allowed range
    num_rand_samples = 40 
    final_lamda =  torch.zeros((grad.shape[0],1))
    
    #Solve optim for each example in the batch
    for idx in range(lamda.shape[0]):
        #Optim corresponding to mahalanobis dis < radius
        if lamda[idx] == 1:
            min_left = np.inf
            best_lam = 0
            for k in range(num_rand_samples):
                val = np.random.uniform(low = lamda_lower_limit[idx], high = 0)
                left_val = check_right_part1(val, grad[idx], diff[idx], radius, device)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val
            
            final_lamda[idx] = best_lam
        
        #Optim corresponding to mahalanobis dis > gamma * radius
        elif lamda[idx] == 2:
            min_left = np.inf
            best_lam = np.inf
            for k in range(num_rand_samples):
                val = np.random.uniform(low = 0, high = nu_upper_limit[idx])
                left_val = check_right_part2(val, grad[idx], diff[idx], radius, device, gamma)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val
            
            final_lamda[idx] = 1.0/best_lam       

        else:
            final_lamda[idx] = 0

    final_lamda = final_lamda.to(device)
    for j in range(diff.shape[0]):
        diff[j,:] = diff[j,:]/(1+final_lamda[j]*grad[j,:])

    return diff

def get_gradients(model, device, data, target):
    data = data.to(torch.float)
    target = target.to(torch.float).squeeze()
    
    data_copy = data.detach().requires_grad_()
    logits = model(data_copy).squeeze(dim=1)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target)
    
    grad = torch.autograd.grad(ce_loss, data_copy)[0]
    return torch.abs(grad)

# ============================================================
# 1) Datasets & Close Negative Generation
# ============================================================
class PacketImageDataset(Dataset):
    def __init__(self, root_dir: str, label: int, img_size: int = 32, augment: bool = False):
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

class CustomDataset(Dataset):
    """ Yakın negatif verilerini tutmak için özel Dataset """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # DROCCTrainer yapısı (data, target, path) beklediği için 3. eleman olarak dummy döner
        return self.data[idx], self.labels[idx], "close_neg_path"

def get_close_negs(test_loader, device, max_samples: int = 50_000):
    """
    Test setindeki NORMAL (label=0) görüntülerde global rastgele piksel maskelemesi
    ile yakın negatif (anomali) veri üretir. Üretilen örnekler attack sınıfı (label=1)
    olarak etiketlenir.

    max_samples: Üretilecek toplam yakın negatif örnek sayısının üst sınırı.
    """
    print(f"\n[INFO] Yakın Negatif (Close Negative) test seti oluşturuluyor (limit={max_samples})...")
    close_neg_data = None
    total = 0

    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.to(torch.float)
        target = target.to(torch.float)

        # Normal sınıfın etiketi 0
        data_0 = data[target == 0]
        if data_0.numel() == 0:
            continue

        # Limite göre bu batch'ten kaç örnek alacağımızı ayarla
        remaining = max_samples - total
        if remaining <= 0:
            break
        if data_0.shape[0] > remaining:
            data_0 = data_0[:remaining]

        aug1 = data_0.clone()
        indices = np.random.choice(
            np.arange(torch.numel(aug1)),
            replace=False,
            size=int(torch.numel(aug1) * 0.4),
        )
        aug1[np.unravel_index(indices, np.shape(aug1))] = torch.min(data)

        if close_neg_data is None:
            close_neg_data = aug1
        else:
            close_neg_data = torch.cat((close_neg_data, aug1), dim=0)

        total += aug1.shape[0]
        if total >= max_samples:
            break

    if close_neg_data is None:
        raise ValueError("Test setinde hiç Normal (0) veri bulunamadı!")

    close_neg_data = close_neg_data.detach().cpu().numpy()
    # Yakın negatifler anomali gibi davranır → attack sınıfı (label=1)
    close_neg_labels = np.ones((close_neg_data.shape[0]))

    print(f"[INFO] {len(close_neg_data)} adet Yakın Negatif örnek başarıyla oluşturuldu.")
    return CustomDataset(close_neg_data, close_neg_labels)


# ============================================================
# 2) Model
# ============================================================

class MNIST_LeNet(nn.Module):
    """
    LeNet tarzı basit mimari (3x3 kernel). `img_size`'a göre pool sayısı
    dinamik: uzamsal boyut >= 2 olduğu sürece en fazla 2 kez pool uygulanır.

        28x28 -> Conv -> BN -> LeakyReLU -> Pool (14) -> Conv -> BN -> LeakyReLU -> Pool (7)
        6x6   -> Conv -> BN -> LeakyReLU -> Pool (3)  -> Conv -> BN -> LeakyReLU         (3)
        Flatten -> FC(flat -> 64)  (embedding)
        FC(64 -> 1)                (logit; BCEWithLogits sigmoid uygular)
    """
    def __init__(self, img_size: int = 6):
        super().__init__()

        self.img_size = img_size
        self.rep_dim  = 64

        self.pool  = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 3, bias=False, padding=1)
        self.bn1   = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 3, bias=False, padding=1)
        self.bn2   = nn.BatchNorm2d(4, eps=1e-04, affine=False)

        # Uzamsal boyut 2'nin altına düşmeyecek şekilde en fazla 2 pool uygula
        s = img_size
        num_pools = 0
        max_pools = 2
        while num_pools < max_pools and s // 2 >= 2:
            s = s // 2
            num_pools += 1
        self.num_pools      = num_pools
        self._final_spatial = s
        self._flatten_dim   = 4 * s * s

        self.fc1 = nn.Linear(self._flatten_dim, self.rep_dim, bias=False)
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


def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
    if epoch < only_ce_epochs:
        lr = learning_rate
    else:
        progress = (epoch - only_ce_epochs) / max(total_epochs - only_ce_epochs, 1)
        lr = learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ============================================================
# 3) DROCCTrainer (HATALAR DÜZELTİLMİŞ VERSİYON)
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
            "train_ce": [], "train_adv": [], "train_compactness": [],
            "train_mean_dist": [], "test_auc": [], "test_acc": [],
        }
        self._best_auc         = -float("inf")
        self._best_state_dict  = None
        self._best_recall_fpr3 = -float("inf")

    def train(self, train_loader, learning_rate, lr_scheduler, total_epochs,
              only_ce_epochs=5, ascent_step_size=0.001, ascent_num_steps=50,
              test_loader=None, closeneg_loader=None):
        
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
                target = target.to(self.device).float().squeeze()

                self.optimizer.zero_grad()

                logits  = self.model(data).squeeze(dim=1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                epoch_ce_loss += ce_loss.item()

                if epoch >= only_ce_epochs:
                    # Normal örnekleri seç (label=0) ve adversarial kayıp hesapla
                    data = data[target == 0]
                    target = torch.zeros(data.shape[0]).to(self.device)
                    gradients = get_gradients(self.model, self.device, data, target)
                    adv_loss = self.one_class_adv_loss(data, gradients)
                    epoch_adv_loss += adv_loss.item()
                    loss            = ce_loss + adv_loss * self.lamda
                else:
                    loss = ce_loss

                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(CE=f"{ce_loss.item():.4f}", Adv=f"{epoch_adv_loss:.4f}")

            n_batches     = batch_idx + 1
            avg_ce        = epoch_ce_loss / n_batches
            avg_adv       = epoch_adv_loss / n_batches
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
                f"Train EmbDist: {mean_d:.4f}±{std_d:.4f}  within_R: {in_r*100:.1f}%"
            )

            if test_loader is not None:
                test_scores, test_labels = self.get_scores_and_labels(test_loader)
                try:
                    test_auc = roc_auc_score(test_labels, test_scores)
                except ValueError:
                    test_auc = float("nan")
                test_preds = (test_scores >= 0.5).astype(int)
                test_acc   = accuracy_score(test_labels, test_preds)

                self.history["test_auc"].append(test_auc)
                self.history["test_acc"].append(test_acc)

                # Varsayılan log: AUC & ACC
                log_msg = f"[TEST] Epoch {epoch+1:>3}/{total_epochs}  |  AUC: {test_auc:.4f}  |  ACC: {test_acc:.4f}"

                # DROCC-LF tarzı FPR@3/5 değerlendirmesi (far + close neg birlikte)
                if closeneg_loader is not None:
                    close_scores, _ = self.get_scores_and_labels(closeneg_loader)
                    # Pozitif sınıf (anomali) = attack = 1
                    pos_scores       = test_scores[test_labels == 1]   # Attack
                    far_neg_scores   = test_scores[test_labels == 0]   # Normal
                    close_neg_scores = close_scores                    # Close neg (fake attack)

                    p3, r3 = cal_precision_recall_lf(pos_scores, far_neg_scores, close_neg_scores, 0.03)
                    p5, r5 = cal_precision_recall_lf(pos_scores, far_neg_scores, close_neg_scores, 0.05)

                    log_msg += f"  |  FPR3: P={p3:.4f}, R={r3:.4f}  FPR5: P={p5:.4f}, R={r5:.4f}"

                    # En iyi modeli FPR@3 altındaki recall'e göre seç
                    if np.isfinite(r3) and r3 > self._best_recall_fpr3:
                        self._best_recall_fpr3 = r3
                        self._best_state_dict  = copy.deepcopy(self.model.state_dict())
                else:
                    # Eğer close-neg yoksa AUC'a göre seç
                    if np.isfinite(test_auc) and test_auc > self._best_auc:
                        self._best_auc        = test_auc
                        self._best_state_dict = copy.deepcopy(self.model.state_dict())

                print(log_msg)

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)
            if self._best_recall_fpr3 > -float("inf"):
                print(f"\n[INFO] Best model restored (Best Recall@FPR3={self._best_recall_fpr3:.4f}).")
            else:
                print(f"\n[INFO] Best model restored (Test AUC={self._best_auc:.4f}).")

    def one_class_adv_loss(self, x_train_data, gradients):
        batch_size = len(x_train_data)
        
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():
                # Adversarial hedef: attack sınıfı = 1 (normal=0 konvansiyonu)
                # Gradient ascent ile adv örnekleri normal manifolduna YAKIN
                # (BCE(logits, target=1) kaybını maksimize eden yön = logit'i
                # düşürmek = model'in "normal" demesini zorlayan hard örnekler)
                new_targets = torch.ones(batch_size, 1).to(self.device).squeeze().float()
                logits    = self.model(x_adv_sampled).squeeze(dim=1)
                new_loss  = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad      = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_normalized = grad / grad_norm

            
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 5==0:
                # Project the normal points to the set N_i(r) based on mahalanobis distance
                h = x_adv_sampled - x_train_data
                h_flat = torch.reshape(h, (h.shape[0], -1))
                gradients_flat = torch.reshape(gradients, (gradients.shape[0], -1))
                #Normalize the gradients 
                gradients_normalized = normalize_grads(gradients_flat)
                #Solve the non-convex 1D optimization
                h_flat = optim_solver(gradients_normalized, h_flat, self.radius, self.device, self.gamma)
                h = torch.reshape(h_flat, h.shape)
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere

        # Adv örnekleri attack sınıfı (1) olarak eğit: anomali sınırını sıkılaştır
        adv_pred = self.model(x_adv_sampled).squeeze(dim=1)
        adv_targets = torch.ones_like(adv_pred)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, adv_targets)

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

        self._last_train_info = {
            "mean_dist" : dists.mean().item(),
            "std_dist"  : dists.std().item(),
            "within_r"  : (dists <= self.radius).float().mean().item(),
        }
        return -dists.mean().item()

    @torch.no_grad()
    def get_scores_and_labels(self, loader):
        self.model.eval()
        scores, labels = [], []
        for data, target, _ in loader:
            logits = self.model(data.to(self.device).float()).squeeze(dim=1)
            scores.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(target.numpy())
        return np.array(scores), np.array(labels)


# ============================================================
# 4) Evaluation & Precision-Recall FPR Metrikleri
# ============================================================
def cal_precision_recall(pos_scores, neg_scores, fpr_target):
    """
    Belirlenen Yanlış Pozitif Oranına (FPR) ulaşıldığında 
    modelin Precision ve Recall değerlerini hesaplar.
    """
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return 0.0, 0.0

    neg_scores = np.asarray(neg_scores).copy()
    num_neg = neg_scores.shape[0]

    # DROCC-LF tarzı: threshold'u NEGATİF skorların (1 - FPR) quantile'ından seç
    neg_scores.sort()
    idx = int(max(0, min((1 - fpr_target) * num_neg, num_neg - 1)))
    thresh = neg_scores[idx]
    
    tp = np.sum(pos_scores > thresh)                 # Anormalliği anormallik olarak bilenler
    fn = np.sum(pos_scores <= thresh)                # Anormalliği normal sananlar
    fp = int(fpr_target * num_neg)                   # Hedef FPR'a göre beklenen FP sayısı
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    return precision, recall


def cal_precision_recall_lf(pos_scores, far_neg_scores, close_neg_scores, fpr):
    """
    DROCC-LF tarzı: uzak ve yakın negatifleri birleştirip
    verilen FPR için precision/recall hesaplar.
    """
    all_neg = np.concatenate((far_neg_scores, close_neg_scores), axis=0)
    num_neg = all_neg.shape[0]
    if num_neg == 0 or pos_scores.shape[0] == 0:
        return 0.0, 0.0

    # Negatif skorları küçükten büyüğe sırala
    all_neg_sorted = np.sort(all_neg.copy())
    idx = int(max(0, min((1 - fpr) * num_neg, num_neg - 1)))
    thresh = all_neg_sorted[idx]

    tp = np.sum(pos_scores > thresh)
    recall = tp / (pos_scores.shape[0] + 1e-10)
    fp = int(fpr * num_neg)
    precision = tp / (tp + fp + 1e-10)
    return precision, recall

def evaluate_and_plot(
    trainer: DROCCTrainer,
    test_loader: DataLoader,
    closeneg_loader: DataLoader,
    save_path: str = "final_model_performance.png",
):
    print("\n" + "=" * 60)
    print("  EVALUATION (STANDARD TEST & CLOSE NEGATIVES)")
    print("=" * 60)

    # 1. Standart Test Verisi Skorları
    # NOT: get_scores_and_labels() çıktısı sigmoid(logit) olup "anomali skoru"dur.
    # Daha büyük skor -> daha anormal (attack).
    test_scores, test_labels = trainer.get_scores_and_labels(test_loader)
    # Pozitif sınıf (anomali) = attack = 1
    pos_scores = test_scores[test_labels == 1]      # Attack (pozitif sınıf)
    far_neg_scores = test_scores[test_labels == 0]  # Normal (uzak negatif)
    
    # 2. Yakın Negatif Test Verisi Skorları
    closeneg_scores, _ = trainer.get_scores_and_labels(closeneg_loader)

    print("\n--- STANDART TEST SETİ METRİKLERİ ---")
    fpr, tpr, roc_thresh = roc_curve(test_labels, test_scores)

    # Youden J indeksi: TPR - FPR maksimum olduğu nokta
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    best_thresh = float(roc_thresh[min(best_idx, len(roc_thresh) - 1)]) if len(roc_thresh) > 0 else 0.5

    preds = (test_scores >= best_thresh).astype(int)
    auc  = roc_auc_score(test_labels, test_scores)
    acc = accuracy_score(test_labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels, preds, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        test_labels, preds, average="weighted", zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        test_labels, preds, labels=[0, 1], zero_division=0
    )
    print(f"Skor yorumu: anomaly score (yüksek=attack/anormal, düşük=normal)")
    print(f"Anomaly score tanımı: anomaly_score = sigmoid(logit)")
    print(f"Genel AUC: {auc:.4f} | Youden Optimal Threshold: {best_thresh:.6f}")
    print(f"Accuracy: {acc:.4f} | Macro P/R/F1: {prec_macro:.4f} / {rec_macro:.4f} / {f1_macro:.4f}")
    # label=0 => Normal, label=1 => Attack
    print(classification_report(test_labels, preds, target_names=["Normal", "Attack"]))

    print("\n--- UZAK NEGATİFLER (Orijinal Attack) İÇİN FPR METRİKLERİ ---")
    p3_far, r3_far = cal_precision_recall(pos_scores, far_neg_scores, 0.03)
    p5_far, r5_far = cal_precision_recall(pos_scores, far_neg_scores, 0.05)
    print(f"Precision @ FPR 3%: {p3_far:.4f} | Recall @ FPR 3%: {r3_far:.4f}")
    print(f"Precision @ FPR 5%: {p5_far:.4f} | Recall @ FPR 5%: {r5_far:.4f}")

    print("\n--- YAKIN NEGATİFLER (Maskelenmiş Anormallikler) İÇİN FPR METRİKLERİ ---")
    p3_close, r3_close = cal_precision_recall(pos_scores, closeneg_scores, 0.03)
    p5_close, r5_close = cal_precision_recall(pos_scores, closeneg_scores, 0.05)
    print(f"Precision @ FPR 3%: {p3_close:.4f} | Recall @ FPR 3%: {r3_close:.4f}")
    print(f"Precision @ FPR 5%: {p5_close:.4f} | Recall @ FPR 5%: {r5_close:.4f}")

    # Grafik Çizimi (üst: 4 panel, alt: metrik özeti)
    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.22], hspace=0.4, wspace=0.28)
    fig.suptitle("DROCC Anomaly Detector Evaluation", fontsize=15, weight="bold", y=0.98)

    metrics_block = (
        f"Score semantics: anomaly_score = sigmoid(logit) — yüksek skor = attack (pozitif sınıf)\n"
        f"Youden threshold = {best_thresh:.4f}  |  AUC = {auc:.4f}  |  Accuracy = {acc:.4f}\n"
        f"Macro:  Precision = {prec_macro:.4f}  |  Recall = {rec_macro:.4f}  |  F1 = {f1_macro:.4f}\n"
        f"Weighted:  Precision = {prec_w:.4f}  |  Recall = {rec_w:.4f}  |  F1 = {f1_w:.4f}\n"
        f"Per class — Normal (0): P={p_cls[0]:.4f}  R={r_cls[0]:.4f}  F1={f1_cls[0]:.4f}  |  "
        f"Attack (1): P={p_cls[1]:.4f}  R={r_cls[1]:.4f}  F1={f1_cls[1]:.4f}"
    )

    # ROC Eğrisi
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={auc:.4f}")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax1.scatter(fpr[best_idx], tpr[best_idx], color="red", zorder=5, label="Youden thresh")
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve (Attack=Positive)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(test_labels, preds)
    ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    thresh_c = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh_c else "black",
            )
    ax2.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion Matrix\n(Youden thresh={best_thresh:.3f})",
    )

    # Skor Dağılımı (Anomaly score; yüksek değer daha anormal/attack)
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 1, 40)
    ax3.hist(far_neg_scores, bins=bins, alpha=0.6, color="steelblue", label="Normal")
    ax3.hist(pos_scores, bins=bins, alpha=0.6, color="tomato", label="Attack (Far)")
    ax3.hist(closeneg_scores, bins=bins, alpha=0.6, color="purple", label="Close Negs (Masked)")
    ax3.axvline(best_thresh, color="black", linestyle="--", label=f"Youden={best_thresh:.2f}")
    ax3.set(xlabel="Anomaly Score (sigmoid logit)", ylabel="Count", title="Score Distribution")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Eğitim Geçmişi
    ax4 = fig.add_subplot(gs[0, 3])
    epochs = range(1, len(trainer.history["train_mean_dist"]) + 1)
    ax4.plot(epochs, trainer.history["train_mean_dist"], "o-", color="darkorange", label="Emb Mean Dist (↓)")
    ax4_r = ax4.twinx()
    ax4_r.plot(epochs, trainer.history["train_ce"], "s--", color="steelblue", label="Train CE")
    ax4.set(xlabel="Epoch", title="Training History")
    ax4.legend(loc="upper left")
    ax4_r.legend(loc="upper right")
    ax4.grid(alpha=0.3)

    axm = fig.add_subplot(gs[1, :])
    axm.axis("off")
    axm.text(
        0.5,
        0.5,
        metrics_block,
        transform=axm.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray", alpha=0.95),
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[SAVED] Figure → {save_path}")
    plt.show()

# ============================================================
# 5) Main
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DROCC Trainer")
    parser.add_argument("--train_normal_dir", default="wustlehms_images_onehot/train/normal")
    parser.add_argument("--test_normal_dir",  default="wustlehms_images_onehot/test/normal")
    parser.add_argument("--test_attack_dir",  default="wustlehms_images_onehot/test/attack")
    parser.add_argument("--img_size",         type=int,   default=6)
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--epochs",           type=int,   default=100)
    parser.add_argument("--only_ce_epochs",   type=int,   default=50)
    parser.add_argument("--lr",               type=float, default=0.001)
    parser.add_argument("--lamda",            type=float, default=0.2)
    parser.add_argument("--radius",           type=float, default=0.5)
    parser.add_argument("--gamma",            type=float, default=2.0)
    parser.add_argument("--ascent_step_size", type=float, default=0.001)
    parser.add_argument("--ascent_num_steps", type=int,   default=50)
    parser.add_argument("--max_closeneg",     type=int,   default=50_000,
                        help="Yakın negatif örnek sayısı üst sınırı")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output_fig",       default="final_model_performance.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {DEVICE}")

    # 1. Veri Yükleyicilerin Hazırlanması
    # Normal = 0, Attack = 1  (standart DROCC konvansiyonu)
    train_ds = PacketImageDataset(args.train_normal_dir, label=0, img_size=args.img_size, augment=True)
    test_ds  = ConcatDataset([
        PacketImageDataset(args.test_normal_dir, label=0, img_size=args.img_size),
        PacketImageDataset(args.test_attack_dir, label=1, img_size=args.img_size),
    ])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Yakın Negatif Test Setinin Üretilmesi (Test Loader kullanılarak)
    closeneg_dataset = get_close_negs(test_loader, DEVICE, max_samples=args.max_closeneg)
    closeneg_loader  = DataLoader(closeneg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. Model ve Eğitmen Tanımlaması
    model = MNIST_LeNet(img_size=args.img_size).to(DEVICE)
    print(
        f"[INFO] Arch: MNIST_LeNet | img={args.img_size} | "
        f"rep_dim={model.rep_dim} | num_pools={model.num_pools} | "
        f"final_spatial={model._final_spatial} | flatten={model._flatten_dim} | "
        f"Params: {sum(p.numel() for p in model.parameters()):,}"
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    trainer = DROCCTrainer(
        model     = model,
        optimizer = optimizer,
        lamda     = args.lamda,
        radius    = args.radius,
        gamma     = args.gamma,
        device    = DEVICE,
    )

    # 4. Eğitim Süreci
    trainer.train(
        train_loader      = train_loader,
        learning_rate     = args.lr,
        lr_scheduler      = adjust_learning_rate,
        total_epochs      = args.epochs,
        only_ce_epochs    = args.only_ce_epochs,
        ascent_step_size  = args.ascent_step_size,
        ascent_num_steps  = args.ascent_num_steps,
        test_loader       = test_loader,
        closeneg_loader   = closeneg_loader,
    )

    # 5. Standart ve Yakın Negatif Değerlendirme
    evaluate_and_plot(
        trainer,
        test_loader,
        closeneg_loader,
        save_path=args.output_fig,
    )

    torch.save(model.state_dict(), "drocc_final_model.pth")
    print("[SAVED] Model weights → drocc_final_model.pth")