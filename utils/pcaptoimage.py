import os
import glob
import numpy as np
from PIL import Image
from scapy.utils import RawPcapReader

# --- YAPILANDIRMA ---
IMAGE_SIZE = 10
MAX_BYTES = IMAGE_SIZE * IMAGE_SIZE # 10x10 = 100 bayt
MAX_PER_CLASS = 250000 # Her sınıf için maksimum üretilecek görsel sayısı (İhtiyaca göre 500000 yapabilirsiniz)

# Kaynak (Input) Dosya ve Klasörleri
TRAIN_PCAP = "datasets/ciciomt/Normal/Normal.pcap"
TEST_NORMAL_PCAP = "datasets/ciciomt/Normal/Benigntest.pcap"
TEST_ATTACK_DIR = "datasets/ciciomt/Malicious/"

# Hedef (Output) Klasörleri
OUT_TRAIN_DIR = "cicpcapimages250k/train/"
OUT_TEST_NORMAL_DIR = "cicpcapimages250k/test/normal/"
OUT_TEST_ATTACK_DIR = "cicpcapimages250k/test/attack/"

def setup_directories():
    """Hedef klasörleri oluşturur."""
    dirs_to_create = [OUT_TRAIN_DIR, OUT_TEST_NORMAL_DIR, OUT_TEST_ATTACK_DIR]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"Klasör hazır: {d}")

def packet_to_image(packet_bytes, save_path):
    """
    Ham baytları 10x10 PNG görsele çevirir.
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

def process_pcap(pcap_path, output_dir, prefix="pkt", global_count=0, file_limit=MAX_PER_CLASS):
    """
    Verilen PCAP dosyasını okur ve SADECE bu dosya için belirlenen limite (file_limit) 
    ulaşana kadar paketleri PNG olarak kaydeder.
    Genel sayacı (global_count) geri döndürür.
    """
    if not os.path.exists(pcap_path):
        print(f"HATA: Dosya bulunamadı -> {pcap_path}")
        return global_count

    print(f"İşleniyor: {pcap_path}")
    print(f" -> Bu dosya için hedef kota: {file_limit} paket")
    
    file_count = 0 # Sadece bu dosyanın içinden okunan paket sayısı
    
    try:
        with RawPcapReader(pcap_path) as pcap_reader:
            for pkt_data, pkt_metadata in pcap_reader:
                # BU DOSYA İÇİN belirlenen kotaya ulaşıldıysa durdur
                if file_count >= file_limit:
                    print(f"  >>> Bu dosya için belirlenen limite ({file_limit}) ulaşıldı. Diğer dosyaya geçiliyor.")
                    break
                
                # Dosya ismi formatı (global_count kullanarak isimlendiriyoruz ki isimler çakışmasın)
                filename = f"{prefix}_{global_count:07d}.png"
                save_path = os.path.join(output_dir, filename)
                
                packet_to_image(pkt_data, save_path)
                file_count += 1
                global_count += 1
                
                # İlerleme durumunu göster
                if file_count % 50000 == 0:
                    print(f"  {file_count} / {file_limit} paket işlendi...")
                    
        print(f"Bu dosyadan toplam çıkarılan: {file_count} (Genel Toplam: {global_count})\n")
    except Exception as e:
        print(f"HATA: {pcap_path} okunurken sorun oluştu: {e}")
        
    return global_count

def main():
    print("--- PCAP to Image Dönüşüm İşlemi Başlıyor ---\n")
    setup_directories()
    
    # 1. Eğitim verisi (Normal)
    print("\n--- 1. Eğitim Verileri (Normal) Çıkarılıyor ---")
    process_pcap(TRAIN_PCAP, OUT_TRAIN_DIR, prefix="train_normal", global_count=0, file_limit=MAX_PER_CLASS)
    
    # 2. Test verisi (Normal)
    print("\n--- 2. Test Verileri (Normal) Çıkarılıyor ---")
    process_pcap(TEST_NORMAL_PCAP, OUT_TEST_NORMAL_DIR, prefix="test_normal", global_count=0, file_limit=MAX_PER_CLASS)
    
    # 3. Test verisi (Saldırı) - EŞİT DAĞILIMLI
    print("\n--- 3. Test Verileri (Attack) Çıkarılıyor ---")
    attack_count = 0
    if os.path.exists(TEST_ATTACK_DIR):
        attack_pcaps = glob.glob(os.path.join(TEST_ATTACK_DIR, "*.pcap"))
        
        if not attack_pcaps:
            print(f"UYARI: {TEST_ATTACK_DIR} klasöründe hiç .pcap dosyası bulunamadı.")
        else:
            num_attack_files = len(attack_pcaps)
            per_file_limit = MAX_PER_CLASS // num_attack_files
            remainder = MAX_PER_CLASS % num_attack_files
            
            print(f"Toplam {num_attack_files} adet farklı Attack PCAP dosyası bulundu.")
            print(f"Genel limit ({MAX_PER_CLASS}), dosyalara eşit bölüştürülüyor: Dosya başı ortalama {per_file_limit} paket.\n")
            
            for i, pcap_file in enumerate(attack_pcaps):
                base_name = os.path.splitext(os.path.basename(pcap_file))[0]
                
                # Tam sayıya bölünememe durumu varsa (Örn: 10000 limit / 3 dosya), artığı son dosyaya ekle
                current_file_limit = per_file_limit + (remainder if i == num_attack_files - 1 else 0)
                
                attack_count = process_pcap(
                    pcap_path=pcap_file, 
                    output_dir=OUT_TEST_ATTACK_DIR, 
                    prefix=f"attack_{base_name}", 
                    global_count=attack_count, 
                    file_limit=current_file_limit
                )
    else:
        print(f"HATA: Saldırı klasörü bulunamadı -> {TEST_ATTACK_DIR}")

    print("\n--- Tüm İşlemler Tamamlandı ---")

if __name__ == "__main__":
    main()