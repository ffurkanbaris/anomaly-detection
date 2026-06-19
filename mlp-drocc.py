import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
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
from sklearn.model_selection import train_test_split

from trainer.drocclftrainer import DROCCLFTrainer, cal_precision_recall

class DROCCModel(nn.Module):

    def __init__(self, input_dim=45):
        super().__init__()
        self.rep_dim = 128
        self.fc1 = nn.Linear(input_dim, self.rep_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(self.rep_dim, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x)) 
        x = self.fc2(x)
        return x

class CustomVectorDataset(Dataset):
    def __init__(self, csv_file, label_value):
        print(f"Yükleniyor: {csv_file}")
        df = pd.read_csv(csv_file)
        
        feature_columns = [c for c in df.columns if c not in ['Label', 'DataType']]
        self.data = df[feature_columns].values
        self.labels = [label_value] * len(self.data)
        self.input_dim = self.data.shape[1] 
        print(f"[{csv_file}] -> {len(self.data)} satır ({self.input_dim} özellik) yüklendi.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vector = self.data[idx]
        tensor_data = torch.tensor(vector, dtype=torch.float32) / 255.0
        label = self.labels[idx]
        return tensor_data, label, torch.tensor([0])

class TensorDatasetWrapper(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], torch.tensor([0])
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

def calculate_and_print_all_metrics(pos_scores, far_neg_scores, fpr_threshold=0.05):
    
    all_neg_scores = np.asanyarray(far_neg_scores)
    
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(all_neg_scores)])
    y_scores = np.concatenate([pos_scores, all_neg_scores])
    
    auc_score = roc_auc_score(y_true, y_scores)
    
    num_neg = all_neg_scores.shape[0]
    idx = int((1 - fpr_threshold) * num_neg)
    sorted_neg = np.sort(all_neg_scores)
    thresh = sorted_neg[idx] if num_neg > 0 else 0.5
    
    y_pred = (y_scores > thresh).astype(int)
    
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

def plot_evaluation_results(pos_scores, far_neg_scores, save_dir, fpr_threshold=0.05):
    all_neg_scores = far_neg_scores
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(all_neg_scores)])
    y_scores = np.concatenate([pos_scores, all_neg_scores])

    plt.figure(figsize=(21, 5))

    num_neg = all_neg_scores.shape[0]
    idx = int((1 - fpr_threshold) * num_neg)
    sorted_neg = np.sort(all_neg_scores)
    fpr_thresh = sorted_neg[idx] if num_neg > 0 else 0.5

    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=1)
    youden_idx = np.argmax(roc_tpr - roc_fpr)
    youden_thresh = roc_thresholds[youden_idx]

    plt.subplot(1, 3, 1)
    plt.hist(pos_scores, bins=50, alpha=0.6, label='Normal Veriler (Pos)', color='green', density=True)
    plt.hist(all_neg_scores, bins=50, alpha=0.6, label='Attack/Neg Veriler', color='red', density=True)
    plt.axvline(
        x=fpr_thresh,
        color='black',
        linestyle='dashed',
        linewidth=2,
        label=f'Threshold (@FPR {fpr_threshold*100}%)'
    )
    plt.axvline(
        x=youden_thresh,
        color='blue',
        linestyle='dotted',
        linewidth=2,
        label='Threshold (Youden J)'
    )
    plt.title('Model Skor Dağılımı (Score Distribution)')
    plt.xlabel('Anomali Skoru (Yüksek = Normal)')
    plt.ylabel('Yoğunluk')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    y_pred_fpr = (y_scores > fpr_thresh).astype(int)
    cm_fpr = confusion_matrix(y_true, y_pred_fpr, labels=[0, 1])

    plt.subplot(1, 3, 2)
    plt.imshow(cm_fpr, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (@ FPR {fpr_threshold*100}%)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Attack (0)', 'Normal (1)'])
    plt.yticks(tick_marks, ['Attack (0)', 'Normal (1)'])

    thresh_cm_fpr = cm_fpr.max() / 2.
    for i in range(cm_fpr.shape[0]):
        for j in range(cm_fpr.shape[1]):
            plt.text(
                j, i, format(cm_fpr[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_fpr[i, j] > thresh_cm_fpr else "black",
                fontsize=12, fontweight='bold'
            )
    plt.ylabel('Gerçek Etiketler (True Label)')
    plt.xlabel('Tahmin Edilen Etiketler (Predicted Label)')

    y_pred_youden = (y_scores > youden_thresh).astype(int)
    cm_youden = confusion_matrix(y_true, y_pred_youden, labels=[0, 1])

    plt.subplot(1, 3, 3)
    plt.imshow(cm_youden, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Youden J)')
    plt.colorbar()
    plt.xticks(tick_marks, ['Attack (0)', 'Normal (1)'])
    plt.yticks(tick_marks, ['Attack (0)', 'Normal (1)'])

    thresh_cm_youden = cm_youden.max() / 2.
    for i in range(cm_youden.shape[0]):
        for j in range(cm_youden.shape[1]):
            plt.text(
                j, i, format(cm_youden[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_youden[i, j] > thresh_cm_youden else "black",
                fontsize=12, fontweight='bold'
            )
    plt.ylabel('Gerçek Etiketler (True Label)')
    plt.xlabel('Tahmin Edilen Etiketler (Predicted Label)')

    save_path = os.path.join(save_dir, 'evaluation_distribution_cm.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Başarılı] Test Değerlendirme Çıktıları (Dağılım & CM) kaydedildi: {save_path}")

def main():
    
    train_normal_csv = "preprocessed_data/train_normal_scaled.csv"
    test_normal_csv = "preprocessed_data/test_normal_scaled.csv"
    test_attack_csv = "preprocessed_data/test_attack_scaled.csv"

    train_dataset = CustomVectorDataset(csv_file=train_normal_csv, label_value=1)
    
    test_normal_dataset = CustomVectorDataset(csv_file=test_normal_csv, label_value=1)
    test_attack_dataset = CustomVectorDataset(csv_file=test_attack_csv, label_value=0)
    test_dataset = torch.utils.data.ConcatDataset([test_normal_dataset, test_attack_dataset])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = DROCCModel(input_dim=train_dataset.input_dim).to(device)
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
        history = trainer.train(train_loader, test_loader, args.lr, adjust_learning_rate, args.epochs,
                      ascent_step_size=args.ascent_step_size, only_ce_epochs=args.only_ce_epochs)
        trainer.save(args.model_dir)
        print("Eğitim tamamlandı ve model kaydedildi.")
        
        plot_training_metrics(history, args.model_dir)
        
        print("\nTest seti üzerinde değerlendirme ve Skor Dağılımı yapılıyor...")
        _, pos_scores, far_neg_scores = trainer.test(test_loader, get_auc=False)

        calculate_and_print_all_metrics(pos_scores, far_neg_scores,  fpr_threshold=0.05)
        plot_evaluation_results(pos_scores, far_neg_scores, args.model_dir, fpr_threshold=0.05)

    else:
        if os.path.exists(os.path.join(args.model_dir, 'model.pt')):
            trainer.load(args.model_dir)
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
            
        _, pos_scores, far_neg_scores = trainer.test(test_loader, get_auc=False)

        calculate_and_print_all_metrics(pos_scores, far_neg_scores, fpr_threshold=0.05)
        
        precision_fpr03, recall_fpr03 = cal_precision_recall(pos_scores, far_neg_scores, 0.03)
        precision_fpr05, recall_fpr05 = cal_precision_recall(pos_scores, far_neg_scores, 0.05)
        
        print('Test Precision @ FPR 3% : {}, Recall @ FPR 3%: {}'.format(precision_fpr03, recall_fpr03))
        print('Test Precision @ FPR 5% : {}, Recall @ FPR 5%: {}'.format(precision_fpr05, recall_fpr05))

        plot_evaluation_results(pos_scores, far_neg_scores, args.model_dir, fpr_threshold=0.05)

if __name__ == '__main__':
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
    parser.add_argument('--radius', type=float, default=0.1, metavar='N', help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1.0, metavar='N', help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N', help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N', help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=0, metavar='N', help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N', help='r to gamma * r projection for the set N_i(r)')
    parser.add_argument('-d', '--data_path', type=str, default='.')
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    main()
