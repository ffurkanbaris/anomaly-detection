import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from trainer.drocclftrainer import DROCCLFTrainer, cal_precision_recall

# ---------------------------------------------------------
# 1. MODEL MİMARİSİ (6x6 Görüntü Boyutuna Göre)
# ---------------------------------------------------------
class DROCCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 1 * 1, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, 1, bias=False)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 7, 7)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# 2. ÖZEL VERİSETİ SINIFI
# ---------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, normal_dir=None, attack_dir=None, transform=None):
        self.transform = transform
        self.filepaths = []
        self.labels = []
        
        if normal_dir and os.path.exists(normal_dir):
            normal_files = glob.glob(os.path.join(normal_dir, "*.png"))
            self.filepaths.extend(normal_files)
            self.labels.extend([1] * len(normal_files))
            
        if attack_dir and os.path.exists(attack_dir):
            attack_files = glob.glob(os.path.join(attack_dir, "*.png"))
            self.filepaths.extend(attack_files)
            self.labels.extend([0] * len(attack_files))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, torch.tensor([0])

class TensorDatasetWrapper(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], torch.tensor([0])

# ---------------------------------------------------------
# 3. YARDIMCI VE ÇİZİM FONKSİYONLARI
# ---------------------------------------------------------
def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
    epoch = epoch - only_ce_epochs
    drocc_epochs = total_epochs - only_ce_epochs
    if epoch <= drocc_epochs:
        lr = learning_rate * 0.01
    if epoch <= 0.80 * drocc_epochs:
        lr = learning_rate * 0.1
    if epoch <= 0.40 * drocc_epochs:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def get_close_negs(test_loader, device):
    batch_idx = -1
    close_neg_data = None
    for data, target, _ in test_loader:
        batch_idx += 1
        data, target = data.to(device), target.to(device)
        data = data.to(torch.float)
        target = target.to(torch.float)
        
        data_0 = data[target == 1]
        if data_0.size(0) == 0: continue
            
        aug1 = data_0.clone()
        indices = np.random.choice(np.arange(torch.numel(aug1)), replace=False,
                                   size=int(torch.numel(aug1) * 0.4))
        aug1[np.unravel_index(indices, np.shape(aug1))] = torch.min(data)
        
        if close_neg_data is None:
            close_neg_data = aug1
        else:
            close_neg_data = torch.cat((close_neg_data, aug1), dim=0)

    close_neg_data = close_neg_data.detach().cpu()
    close_neg_labels = torch.zeros(close_neg_data.shape[0])
    
    return TensorDatasetWrapper(close_neg_data, close_neg_labels)


def calculate_and_print_all_metrics(pos_scores, far_neg_scores, close_neg_scores, fpr_threshold=0.05):
    
    all_neg_scores = np.concatenate((far_neg_scores, close_neg_scores), axis=0)
    
    # Gerçek Etiketler (Normal = 1, Attack = 0) ve Skorları Birleştirme
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(all_neg_scores)])
    y_scores = np.concatenate([pos_scores, all_neg_scores])
    
    # 1. Eşikten Bağımsız Metrik: ROC-AUC Hesaplama
    auc_score = roc_auc_score(y_true, y_scores)
    
    # 2. Threshold Belirleme (Hedeflenen FPR'ye göre eşik noktası kesimi)
    num_neg = all_neg_scores.shape[0]
    idx = int((1 - fpr_threshold) * num_neg)
    sorted_neg = np.sort(all_neg_scores)
    thresh = sorted_neg[idx] if num_neg > 0 else 0.5
    
    # Eşiğe Göre Sınıflandırma Kararı (Eşikten büyükse 1, küçükse 0)
    y_pred = (y_scores > thresh).astype(int)
    
    # Diğer Metrikler
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    print("\n" + "="*50)
    print("="*50)
    print(f"   ➤ ROC-AUC Skoru : {auc_score:.4f} (1.0 = Kusursuz, 0.5 = Rastgele)")
    print("-" * 50)
    print(f"🔹 Seçilen Eşik (@FPR {fpr_threshold*100}% Threshold = {thresh:.4f}) İçin:")
    print(f"   ➤ Accuracy (Doğruluk) : {acc:.4f}")
    print(f"   ➤ Precision (Kesinlik): {prec:.4f}")
    print(f"   ➤ Recall (Duyarlılık) : {rec:.4f}")
    print(f"   ➤ F1-Score            : {f1:.4f}")
    print("-" * 50)
    print("🔹 Confusion Matrix (Karmaşıklık Matrisi):")
    print(f"   [ TN: {cm[0,0]:<5} | FP: {cm[0,1]:<5} ] -> Saldırı (0)")
    print(f"   [ FN: {cm[1,0]:<5} | TP: {cm[1,1]:<5} ] -> Normal (1)")
    print("="*50 + "\n")

def plot_training_metrics(history, save_dir):
    """Eğitim sonu loss ve metrik grafiklerini çizer."""
    epochs = range(1, len(history['ce_loss']) + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['ce_loss'], label='CE Loss', color='blue')
    plt.plot(epochs, history['adv_loss'], label='Adv Loss', color='red')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['precision_fpr03'], label='Precision (FPR 3%)', color='green')
    plt.plot(epochs, history['recall_fpr03'], label='Recall (FPR 3%)', color='purple')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'training_history_plot.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n[Başarılı] Eğitim grafikleri kaydedildi: {save_path}")

def plot_evaluation_results(pos_scores, far_neg_scores, close_neg_scores, save_dir, fpr_threshold=0.05):
    """Test veri seti üzerindeki Skor Dağılımını ve Confusion Matrix'i çizer."""
    all_neg_scores = np.concatenate((far_neg_scores, close_neg_scores), axis=0)
    
    plt.figure(figsize=(14, 5))
    
    # 1. Skor Dağılımı Histogramı (Score Distribution)
    plt.subplot(1, 2, 1)
    plt.hist(pos_scores, bins=50, alpha=0.6, label='Normal Veriler (Pos)', color='green', density=True)
    plt.hist(all_neg_scores, bins=50, alpha=0.6, label='Attack/Neg Veriler', color='red', density=True)
    plt.title('Model Skor Dağılımı (Score Distribution)')
    plt.xlabel('Anomali Skoru (Yüksek = Normal)')
    plt.ylabel('Yoğunluk')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Eşik Değerini (Threshold) Belirleme (FPR = %5 'e göre)
    num_neg = all_neg_scores.shape[0]
    idx = int((1 - fpr_threshold) * num_neg)
    sorted_neg = np.sort(all_neg_scores)
    thresh = sorted_neg[idx] if num_neg > 0 else 0.5
    
    # Threshold çizgisi çizdirme
    plt.axvline(x=thresh, color='black', linestyle='dashed', linewidth=2, label=f'Threshold (@FPR {fpr_threshold*100}%)')
    plt.legend()

    # 2. Karmaşıklık Matrisi (Confusion Matrix)
    # Tahminler: Eşikten büyükse Normal (1), küçükse Attack (0)
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(all_neg_scores)])
    y_pred = np.concatenate([(pos_scores > thresh).astype(int), (all_neg_scores > thresh).astype(int)])
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.subplot(1, 2, 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (@ FPR {fpr_threshold*100}%)')
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Attack (0)', 'Normal (1)'])
    plt.yticks(tick_marks, ['Attack (0)', 'Normal (1)'])
    
    # Kutuların içine sayıları yazdırma
    thresh_cm = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh_cm else "black",
                     fontsize=12, fontweight='bold')
                     
    plt.ylabel('Gerçek Etiketler (True Label)')
    plt.xlabel('Tahmin Edilen Etiketler (Predicted Label)')

    save_path = os.path.join(save_dir, 'evaluation_distribution_cm.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Başarılı] Test Değerlendirme Çıktıları (Dağılım & CM) kaydedildi: {save_path}")

# ---------------------------------------------------------
# 4. ANA DÖNGÜ (MAIN)
# ---------------------------------------------------------
def main():
    transform = transforms.Compose([
        transforms.Resize((7, 7)),
        transforms.ToTensor()
    ])

    train_normal_dir = "network_traffic_7x7_images/train/normal"
    test_normal_dir = "network_traffic_7x7_images/test/normal"
    test_attack_dir = "network_traffic_7x7_images/test/attack"

    train_dataset = CustomImageDataset(normal_dir=train_normal_dir, transform=transform)
    test_dataset = CustomImageDataset(normal_dir=test_normal_dir, attack_dir=test_attack_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    closeneg_test_data = get_close_negs(test_loader, device)
    closeneg_test_loader = DataLoader(closeneg_test_data, batch_size=args.batch_size, shuffle=True)

    model = DROCCModel().to(device)
    model = nn.DataParallel(model)

    if args.optim == 1:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print("using Adam")

    trainer = DROCCLFTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device)

    if args.eval == 0:
        print("Eğitim başlatılıyor...")
        history = trainer.train(train_loader, test_loader, closeneg_test_loader, args.lr, adjust_learning_rate, args.epochs,
                      ascent_step_size=args.ascent_step_size, only_ce_epochs=args.only_ce_epochs)
        trainer.save(args.model_dir)
        print("Eğitim tamamlandı ve model kaydedildi.")
        
        # Eğitim grafikleri
        plot_training_metrics(history, args.model_dir)
        
        # Test Seti üzerinden Skor Dağılımı ve CM grafikleri (Eğitim sonrası)
        print("\nTest seti üzerinde değerlendirme ve Skor Dağılımı yapılıyor...")
        _, pos_scores, far_neg_scores = trainer.test(test_loader, get_auc=False)
        _, _, close_neg_scores = trainer.test(closeneg_test_loader, get_auc=False)

        calculate_and_print_all_metrics(pos_scores, far_neg_scores, close_neg_scores, fpr_threshold=0.05)
        plot_evaluation_results(pos_scores, far_neg_scores, close_neg_scores, args.model_dir, fpr_threshold=0.05)

    else:
        if os.path.exists(os.path.join(args.model_dir, 'model.pt')):
            trainer.load(args.model_dir)
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
            
        _, pos_scores, far_neg_scores = trainer.test(test_loader, get_auc=False)
        _, _, close_neg_scores = trainer.test(closeneg_test_loader, get_auc=False)

        calculate_and_print_all_metrics(pos_scores, far_neg_scores, close_neg_scores, fpr_threshold=0.05)
        
        precision_fpr03, recall_fpr03 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.03)
        precision_fpr05, recall_fpr05 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.05)
        
        print('Test Precision @ FPR 3% : {}, Recall @ FPR 3%: {}'.format(precision_fpr03, recall_fpr03))
        print('Test Precision @ FPR 5% : {}, Recall @ FPR 5%: {}'.format(precision_fpr05, recall_fpr05))

        # Test Seti üzerinden Skor Dağılımı ve CM grafikleri (Sadece Değerlendirme modunda)
        plot_evaluation_results(pos_scores, far_neg_scores, close_neg_scores, args.model_dir, fpr_threshold=0.05)

if __name__ == '__main__':
    """"wustlehms_images --> r = 0.1
        wustlehms_images_onehot --> r = 0.1
    
    """
    torch.set_printoptions(precision=5)
    parser = argparse.ArgumentParser(description='PyTorch DROCC Training with Custom PNG Data')
    parser.add_argument('--normal_class', type=int, default=0, metavar='N', help='Normal class index')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('-oce', '--only_ce_epochs', type=int, default=3, metavar='N', help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=50, metavar='N', help='Number of gradient ascent steps')                        
    parser.add_argument('--hd', type=int, default=128, metavar='N', help='Num hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR', help='step size of gradient ascent')                        
    parser.add_argument('--mom', type=float, default=0.99, metavar='M', help='momentum')
    parser.add_argument('--model_dir', default='log', help='path where to save checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N', help='adv loss to be used or not, 1:use 0:not use(only CE)')
    parser.add_argument('--radius', type=float, default=0.05, metavar='N', help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1, metavar='N', help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N', help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N', help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=0, metavar='N', help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N', help='r to gamma * r projection for the set N_i(r)')
    parser.add_argument('-d', '--data_path', type=str, default='.')
    args = parser.parse_args()

    # Settings
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    main()