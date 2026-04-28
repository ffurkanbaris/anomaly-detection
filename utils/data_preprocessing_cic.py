"""
Etiketsiz Ağ Trafiği Veri Setini Resim Formatına (0-255) Hazırlama
==============================================================
Bu script, etiket sütunu bulunmayan CSV dosyalarını klasör 
mantığına göre birleştirip etiketler (Normal=1, Attack=0) 
ve verileri normalize eder.
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# --- KONFİGÜRASYON ---
CONFIG = {
    # Etiketsiz ham CSV dosyalarınızın yolları
    'train_normal_csv': './datasets/csv/train/Benign_train.pcap.csv',
    'test_normal_csv': './datasets/csv/test/benign/Benign_test.pcap.csv',
    'test_attack_dir': './datasets/csv/test/attack', # Bu klasördeki tüm csv'ler Attack sayılır
    
    'output_dir': './preprocessed_data',
    'random_state': 42,

    # --- SÜTUN KATEGORİLERİ ---
    'drop_features': [],

    # Çok büyük sayılar barındıran ve logaritması alınacak sütunlar
    'log_features': [
        'Duration', 'Rate', 'Srate', 'Drate',
        'Tot sum', 'AVG', 'Std', 'Tot size',
        'IAT', 'Magnitue', 'Radius', 'Covariance', 'Variance'
    ],

    # Zaten 0 ile 1 arasında olan Oran/Bayrak özellikleri
    'rate_features': [
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
        'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC'
    ],

    # Doğrudan Min-Max ölçekleyiciye (0-255) girecek diğer özellikler
    'standard_features': [
        'Header_Length', 'Protocol Type', 'ack_count', 'syn_count', 
        'fin_count', 'rst_count', 'Min', 'Max', 'Number', 'Weight'
    ]
}

def load_and_label_data():
    """
    Dosyaları okur ve geldikleri yere göre 'Label' (Normal=1, Attack=0) ekler.
    Tüm verileri birleştirip (concat) geri döndürür.
    """
    all_dataframes = []
    
    # 1. Eğitim - Normal Veriler (Label = 1)
    if os.path.exists(CONFIG['train_normal_csv']):
        print(f"Yükleniyor (Train Normal): {CONFIG['train_normal_csv']}")
        df_train = pd.read_csv(CONFIG['train_normal_csv'])
        df_train['Label'] = 1
        df_train['DataType'] = 'train' # Ayırmak için gizli sütun
        all_dataframes.append(df_train)
        
    # 2. Test - Normal Veriler (Label = 1)
    if os.path.exists(CONFIG['test_normal_csv']):
        print(f"Yükleniyor (Test Normal): {CONFIG['test_normal_csv']}")
        df_test_norm = pd.read_csv(CONFIG['test_normal_csv'])
        df_test_norm['Label'] = 1
        df_test_norm['DataType'] = 'test_normal'
        all_dataframes.append(df_test_norm)

    # 3. Test - Saldırı Verileri (Label = 0)
    if os.path.exists(CONFIG['test_attack_dir']):
        attack_csvs = glob.glob(os.path.join(CONFIG['test_attack_dir'], "*.csv"))
        for csv_file in attack_csvs:
            print(f"Yükleniyor (Test Attack): {csv_file}")
            df_attack = pd.read_csv(csv_file)
            df_attack['Label'] = 0
            df_attack['DataType'] = 'test_attack'
            all_dataframes.append(df_attack)

    if not all_dataframes:
        raise ValueError("Hiç veri okunamadı! Lütfen dosya yollarını kontrol edin.")

    # Tüm tabloları alt alta birleştir
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n[OK] Veriler birlestirildi. Toplam boyut: {combined_df.shape}")
    
    return combined_df

def clean_data(df):
    """Sütun isimlerindeki boşlukları temizler; sayısal sütunlarda inf/NaN'i güvenli değere çeker."""
    df.columns = df.columns.str.strip()
    if CONFIG['drop_features']:
        drop_cols = [c for c in CONFIG['drop_features'] if c in df.columns]
        df = df.drop(columns=drop_cols)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def apply_scaling(df):
    """Sütunları kendi kategorilerine göre 0-255 aralığına çeker."""
    df_scaled = df.copy()
    
    # Label ve DataType sütunları işleme girmeyecek
    exclude_cols = ['Label', 'DataType']
    
    # 1. Logaritmik Sütunlar (x+1 <= 0 iken log10 NaN/inf üretmesin diye alt sınır clip)
    for col in CONFIG['log_features']:
        if col in df_scaled.columns and col not in exclude_cols:
            raw = pd.to_numeric(df_scaled[col], errors='coerce').replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0).to_numpy(dtype=np.float64)
            x = np.clip(raw + 1.0, 1e-12, None)
            df_scaled[col] = np.log10(x)
            scaler = MinMaxScaler(feature_range=(0, 255))
            df_scaled[col] = scaler.fit_transform(df_scaled[[col]]).flatten()

    # 2. Oran (Rate) Sütunları
    for col in CONFIG['rate_features']:
        if col in df_scaled.columns and col not in exclude_cols:
            df_scaled[col] = np.clip(df_scaled[col] * 255, 0, 255)

    # 3. Standart Sütunlar
    for col in CONFIG['standard_features']:
        if col in df_scaled.columns and col not in exclude_cols:
            scaler = MinMaxScaler(feature_range=(0, 255))
            df_scaled[col] = scaler.fit_transform(df_scaled[[col]]).flatten()

    # Değerleri Tam sayıya (Int) yuvarla; kalan NaN/inf (ör. uç ölçek) 0–255'e sıkıştır
    for col in df_scaled.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_scaled[col]):
            v = np.round(pd.to_numeric(df_scaled[col], errors='coerce').to_numpy(dtype=np.float64))
            v = np.nan_to_num(v, nan=0.0, posinf=255.0, neginf=0.0)
            v = np.clip(v, 0, 255)
            df_scaled[col] = v.astype(np.int64)

    return df_scaled

def save_split_data(df):
    """DataType sütununa bakarak verileri tekrar bölüp kaydeder."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # DataType sütununa göre böl
    train_df = df[df['DataType'] == 'train'].drop(columns=['DataType'])
    test_normal_df = df[df['DataType'] == 'test_normal'].drop(columns=['DataType'])
    test_attack_df = df[df['DataType'] == 'test_attack'].drop(columns=['DataType'])
    
    # Dosyaları kaydet
    train_path = os.path.join(CONFIG['output_dir'], 'train_normal_scaled.csv')
    test_norm_path = os.path.join(CONFIG['output_dir'], 'test_normal_scaled.csv')
    test_attack_path = os.path.join(CONFIG['output_dir'], 'test_attack_scaled.csv')
    
    train_df.to_csv(train_path, index=False)
    test_normal_df.to_csv(test_norm_path, index=False)
    test_attack_df.to_csv(test_attack_path, index=False)
    
    print(f"[OK] Kaydedildi (Train Normal): {train_path} ({len(train_df)} satir)")
    print(f"[OK] Kaydedildi (Test Normal): {test_norm_path} ({len(test_normal_df)} satir)")
    print(f"[OK] Kaydedildi (Test Attack): {test_attack_path} ({len(test_attack_df)} satir)")

def main():
    print("=" * 60)
    print("Etiketsiz Verileri Birleştirme & Normalize Etme (0-255)")
    print("=" * 60)
    
    try:
        # 1. Yükle ve Etiketle (Hepsi tek tablo olur)
        df = load_and_label_data()
        
        # 2. Temizle
        df = clean_data(df)
        
        # 3. Tüm veriler bir aradayken Normalize et (ÖNEMLİ!)
        print("\nTüm veriler üzerinden Min-Max hesaplanıyor...")
        df = apply_scaling(df)
        
        # 4. Verileri tekrar ait oldukları 3 parçaya ayırıp kaydet
        print("\nVeriler ayrılarak kaydediliyor...")
        save_split_data(df)
        
        print("\n[OK] ISLEM TAMAMLANDI!")
        
    except Exception as e:
        print(f"\n[ERROR] Hata olustu: {str(e)}")

if __name__ == "__main__":
    main()