import os
import glob
import numpy as np
from PIL import Image
from scapy.utils import RawPcapReader

# --- YAPILANDIRMA ---
IMAGE_SIZE = 10
MAX_BYTES = IMAGE_SIZE * IMAGE_SIZE # 10x10 = 100 bayt
MAX_PER_CLASS = 1_000_000 # Her sınıf için maksimum üretilecek görsel sayısı

# Kaynak (Input) Dosya ve Klasörleri
TRAIN_PCAP = "datasets/ciciomt/Normal/Normal.pcap"
TEST_NORMAL_PCAP = "datasets/ciciomt/Normal/Benigntest.pcap"
TEST_ATTACK_DIR = "datasets/ciciomt/Malicious/"

# Hedef (Output) Klasörleri
OUT_TRAIN_DIR = "cicpcapimages/train/"
OUT_TEST_NORMAL_DIR = "cicpcapimages/test/normal/"
OUT_TEST_ATTACK_DIR = "cicpcapimages/test/attack/"

def setup_directories():
    """Hedef klasörleri oluşturur."""
    dirs_to_create = [OUT_TRAIN_DIR, OUT_TEST_NORMAL_DIR, OUT_TEST_ATTACK_DIR]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"Klasör hazır: {d}")

def packet_to_image(packet_bytes, save_path):
    """
    Makaledeki yönteme göre ham baytları 10x10 PNG görsele çevirir.
    """
    byte_list = list(packet_bytes)
    
    # Boyutlandırma (Kırpma veya Sıfır ile Doldurma)
    if len(byte_list) >= MAX_BYTES:
        byte_list = byte_list[:MAX_BYTES]
    else:
        byte_list.extend([0] * (MAX_BYTES - len(byte_list)))
        
    # Numpy matrisine (10x10) çevirme
    img_array = np.array(byte_list, dtype=np.uint8).reshape((IMAGE_SIZE, IMAGE_SIZE))
    
    # Matrisi görsele çevirip kaydetme
    img = Image.fromarray(img_array, mode='L')
    img.save(save_path)

def process_pcap(pcap_path, output_dir, prefix="pkt", current_count=0, max_limit=MAX_PER_CLASS):
    """
    Verilen PCAP dosyasını okur ve belirlenen limite ulaşana kadar paketleri PNG olarak kaydeder.
    Kaldığı sayacı (current_count) geri döndürür.
    """
    if not os.path.exists(pcap_path):
        print(f"HATA: Dosya bulunamadı -> {pcap_path}")
        return current_count

    print(f"İşleniyor: {pcap_path} -> {output_dir}")
    
    count = current_count
    try:
        with RawPcapReader(pcap_path) as pcap_reader:
            for pkt_data, pkt_metadata in pcap_reader:
                # Maksimum sınıra ulaşıldıysa işlemi durdur
                if count >= max_limit:
                    print(f"  >>> Maksimum sınır ({max_limit}) ulaşıldı! PCAP okuması durduruluyor.")
                    break
                
                # Dosya ismi formatı (Örn: train_normal_0000001.png)
                filename = f"{prefix}_{count:07d}.png"
                save_path = os.path.join(output_dir, filename)
                
                packet_to_image(pkt_data, save_path)
                count += 1
                
                # İlerleme durumunu göster
                if count % 50000 == 0:
                    print(f"  {count} / {max_limit} paket işlendi...")
                    
        print(f"Bu dosyadan sonra toplam işlenen: {count}\n")
    except Exception as e:
        print(f"HATA: {pcap_path} okunurken sorun oluştu: {e}")
        
    return count

def main():
    print("--- PCAP to Image Dönüşüm İşlemi Başlıyor ---\n")
    setup_directories()
    
    # 1. Eğitim verisi (Normal) - Max 1 Milyon
    process_pcap(TRAIN_PCAP, OUT_TRAIN_DIR, prefix="train_normal", current_count=0, max_limit=MAX_PER_CLASS)
    
    # 2. Test verisi (Normal) - Max 1 Milyon
    process_pcap(TEST_NORMAL_PCAP, OUT_TEST_NORMAL_DIR, prefix="test_normal", current_count=0, max_limit=MAX_PER_CLASS)
    
    # 3. Test verisi (Saldırı) - Tüm dosyalar toplamında Max 1 Milyon
    attack_count = 0
    if os.path.exists(TEST_ATTACK_DIR):
        attack_pcaps = glob.glob(os.path.join(TEST_ATTACK_DIR, "*.pcap"))
        if not attack_pcaps:
            print(f"UYARI: {TEST_ATTACK_DIR} klasöründe hiç .pcap dosyası bulunamadı.")
            
        for pcap_file in attack_pcaps:
            # Eğer önceki dosyalarda zaten 1 milyona ulaştıysak döngüyü kır
            if attack_count >= MAX_PER_CLASS:
                print("Saldırı (Attack) sınıfı için genel limite ulaşıldı. Diğer pcap dosyaları atlanıyor.")
                break
                
            base_name = os.path.splitext(os.path.basename(pcap_file))[0]
            
            # Kaldığı sayacı gönderip, yeni sayacı geri alıyoruz (toplam count takibi için)
            attack_count = process_pcap(
                pcap_path=pcap_file, 
                output_dir=OUT_TEST_ATTACK_DIR, 
                prefix=f"attack_{base_name}", 
                current_count=attack_count, 
                max_limit=MAX_PER_CLASS
            )
    else:
        print(f"HATA: Saldırı klasörü bulunamadı -> {TEST_ATTACK_DIR}")

    print("\n--- Tüm İşlemler Tamamlandı ---")

if __name__ == "__main__":
    main()