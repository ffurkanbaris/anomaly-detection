"""
CSV -> 7x7 Grayscale Image Generator
====================================

Bu script, bir önceki aşamada hazırlanmış ve bölünmüş olan 3 ayrı CSV 
dosyasını (Train Normal, Test Normal, Test Attack) okur. 
Her bir satırdaki 45 özelliği alır, 4 adet 0 (padding) ekleyerek 49'a tamamlar 
ve 7x7 piksel boyutunda grayscale (Siyah-Beyaz) PNG resimlerine dönüştürür.

Çıktı Mimarisi:
    <img_output_path>/
        train/normal/sample_XXXXX_normal.png
        test/normal/sample_XXXXX_normal.png
        test/attack/sample_XXXXX_attack.png
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

# --- YAPILANDIRMA ---
IMG_SIZE = 7
TARGET_FEATURES = IMG_SIZE * IMG_SIZE  # 7x7 = 49 piksel

# Bir önceki aşamadan (preprocessed_data) çıkan CSV yolları
TRAIN_NORMAL_CSV = 'preprocessed_data/train_normal_scaled.csv'
TEST_NORMAL_CSV = 'preprocessed_data/test_normal_scaled.csv'
TEST_ATTACK_CSV = 'preprocessed_data/test_attack_scaled.csv'

# Resimlerin kaydedileceği ana klasör (DROCC modeliniz burayı okuyacak)
IMG_OUTPUT_PATH = 'network_traffic_7x7_images'

def _adjust_row_length(row, target_len=TARGET_FEATURES):
    """Satırı hedef uzunluğa (49) uyması için sıfırla doldurur veya keser."""
    if len(row) == target_len:
        return row
    if len(row) < target_len:
        # Kalan kısımları (Örn: 45 veri varsa son 4 pikseli) 0 ile doldur (Padding)
        return np.append(row, np.zeros(target_len - len(row)))
    # Eğer veri 49'dan büyükse kes (Truncate)
    return row[:target_len]

def _row_to_image(row):
    """Tek bir satır verisini 7x7 PIL Image formatına dönüştürür."""
    adjusted = _adjust_row_length(row)
    # Verileri 0-255 arasında ve 8-bit tamsayı (uint8) olarak sınırla
    pixels = np.clip(adjusted, 0, 255).astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE)
    # L modu (Luminance) siyah-beyaz görüntü demektir
    img = Image.fromarray(pixels, mode='L')
    return img

def process_and_save_csv(csv_path, split_name, class_name, output_root):
    """
    Belirtilen CSV'yi okur, resme çevirir ve ilgili klasöre kaydeder.
    Örn: split_name='train', class_name='normal' -> train/normal/ içine kaydeder.
    """
    if not os.path.exists(csv_path):
        print(f"HATA: Dosya bulunamadı -> {csv_path}")
        return 0

    print(f"\nİşleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 'Label' sütununu çıkararak sadece sayısal özellikleri (pikselleri) alıyoruz
    feature_columns = [c for c in df.columns if c != 'Label']
    features_array = df[feature_columns].values
    
    total_rows = len(features_array)
    if total_rows == 0:
        print("Uyarı: Dosya boş.")
        return 0
        
    print(f"Toplam özellik (sütun) sayısı: {len(feature_columns)}")
    print(f"Dönüştürülecek hedef boyut   : {TARGET_FEATURES} ({IMG_SIZE}x{IMG_SIZE})")
    
    # Hedef klasörü oluştur
    out_dir = os.path.join(output_root, split_name, class_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Her satırı resme dönüştürüp kaydet
    for i, row in enumerate(features_array):
        # Dosya ismi: Örn. sample_00123_normal.png
        filename = os.path.join(out_dir, f"sample_{i:06d}_{class_name}.png")
        img = _row_to_image(row)
        img.save(filename, format='PNG', optimize=True)
        
        # İlerleme durumu yazdır
        if (i + 1) % 50000 == 0 or (i + 1) == total_rows:
            print(f"    [{split_name}/{class_name}] {i + 1} / {total_rows} resim kaydedildi.")
            
    return total_rows

def main():
    print("=" * 60)
    print(f"CSV'den {IMG_SIZE}x{IMG_SIZE} Resim Üretme İşlemi Başlıyor...")
    print("=" * 60)
    
    os.makedirs(IMG_OUTPUT_PATH, exist_ok=True)
    
    # 1. Eğitim - Normal
    train_normal_count = process_and_save_csv(TRAIN_NORMAL_CSV, split_name='train', class_name='normal', output_root=IMG_OUTPUT_PATH)
    
    # 2. Test - Normal
    test_normal_count = process_and_save_csv(TEST_NORMAL_CSV, split_name='test', class_name='normal', output_root=IMG_OUTPUT_PATH)
    
    # 3. Test - Saldırı
    test_attack_count = process_and_save_csv(TEST_ATTACK_CSV, split_name='test', class_name='attack', output_root=IMG_OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("✓ TÜM RESİMLER BAŞARIYLA OLUŞTURULDU!")
    print(f"  -> {IMG_OUTPUT_PATH}/train/normal/ : {train_normal_count} resim")
    print(f"  -> {IMG_OUTPUT_PATH}/test/normal/  : {test_normal_count} resim")
    print(f"  -> {IMG_OUTPUT_PATH}/test/attack/  : {test_attack_count} resim")
    print("\nDROCC Modeli eğitimi için her şey hazır.")

if __name__ == "__main__":
    main()