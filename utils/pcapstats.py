"""Paket boyutu dagilim analizi ve goruntu boyutu hesaplama.

Makale: Golubev et al., "Image-Based Approach to Intrusion Detection", Information 2022.

- Normal: datasets/ciciomt/Normal/Normal.pcap
- Saldiri: datasets/ciciomt/Malicious/*.pcap

Hesaplanan istatistikler (Tablo 1):
    Mode, Median, 80. Persentil, 99. Persentil, Min, Max

Goruntu boyutu (Tablo 4 / Formula 2):
    Simage = ceil(e^(ln(Pstat) / 2))
"""

import glob
import math
import os
import struct

import matplotlib
matplotlib.use("Agg")  # Ekran gerektirmez; sadece dosyaya kaydeder
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
NORMAL_PCAP = "datasets/ciciomt/Normal/Normal.pcap"
MALICIOUS_DIR = "datasets/ciciomt/Malicious"
OUTPUT_PLOT = "pcap_packet_length_distribution.png"

MAX_PACKETS = 2_000_000  # Bellek korumasi icin maksimum paket


# ---------------------------------------------------------------------------
# Hizli PCAP okuyucu (sadece paket basliklarini okur, payload parse etmez)
# ---------------------------------------------------------------------------
# PCAP (classic) magic numbers
_PCAP_MAGIC_LE    = 0xA1B2C3D4
_PCAP_MAGIC_BE    = 0xD4C3B2A1
_PCAP_MAGIC_LE_NS = 0xA1B23C4D
_PCAP_MAGIC_BE_NS = 0x4D3CB2A1

# PCAPNG block types
_BLK_SHB = 0x0A0D0D0A   # Section Header Block
_BLK_IDB = 0x00000001   # Interface Description Block
_BLK_EPB = 0x00000006   # Enhanced Packet Block
_BLK_SPB = 0x00000003   # Simple Packet Block
_BLK_OPB = 0x00000002   # Obsolete Packet Block

# pcapng SHB byte-order magic
_NG_MAGIC_LE = 0x1A2B3C4D
_NG_MAGIC_BE = 0x4D3C2B1A


def _pad32(n: int) -> int:
    """n'i 4'e kat olacak sekilde yukari yuvarla."""
    return (n + 3) & ~3


def _read_pcap(f, max_packets: int, endian: str) -> list[int]:
    """Classic PCAP formatindan orig_len degerlerini oku (basliktan sonra cagirilir)."""
    pkt_struct = struct.Struct(f"{endian}IIII")  # ts_sec ts_usec incl_len orig_len
    lengths = []
    while len(lengths) < max_packets:
        hdr = f.read(16)
        if len(hdr) < 16:
            break
        _, _, incl_len, orig_len = pkt_struct.unpack(hdr)
        lengths.append(orig_len)
        f.seek(incl_len, 1)
    return lengths


def _read_pcapng(f, max_packets: int) -> list[int]:
    """PCAPNG formatindan orig_len degerlerini oku (dosyanin basindayken cagirilir).

    Her blok icin: blk_start + blk_total_len konumuna seek ederek atlama yapilir.
    Bu yaklasim, blok icindeki okuma miktarindan bagimsiz olarak guvenlidir.
    """
    lengths = []
    endian = "<"  # varsayilan; SHB byte-order magic'ten guncellenir

    while len(lengths) < max_packets:
        blk_start = f.tell()

        # Blok tipi (4 bayt)
        type_bytes = f.read(4)
        if len(type_bytes) < 4:
            break
        blk_type = struct.unpack_from("<I", type_bytes)[0]

        # Blok toplam uzunlugu (4 bayt)
        len_bytes = f.read(4)
        if len(len_bytes) < 4:
            break
        blk_total_len = struct.unpack_from(endian + "I", len_bytes)[0]

        if blk_total_len < 12:
            # Gecersiz blok uzunlugu; devam etmek guvenli degil
            break

        if blk_type == _BLK_SHB:
            # Byte-Order Magic'i oku ve endian'i belirle
            bo_bytes = f.read(4)
            if len(bo_bytes) < 4:
                break
            bo_magic = struct.unpack_from("<I", bo_bytes)[0]
            endian = ">" if bo_magic == _NG_MAGIC_BE else "<"
            # blk_total_len'i dogru endian ile yeniden isle
            blk_total_len = struct.unpack_from(endian + "I", len_bytes)[0]

        elif blk_type == _BLK_EPB:
            # Interface ID(4) + TS High(4) + TS Low(4) + cap_len(4) + orig_len(4)
            epb_fixed = f.read(20)
            if len(epb_fixed) < 20:
                break
            _cap_len, orig_len = struct.unpack_from(endian + "II", epb_fixed, 12)
            lengths.append(orig_len)

        elif blk_type == _BLK_OPB:
            # iface(2) + drops(2) + ts_sec(4) + ts_usec(4) + cap_len(4) + orig_len(4)
            opb_fixed = f.read(16)
            if len(opb_fixed) < 16:
                break
            orig_len = struct.unpack_from(endian + "I", opb_fixed, 12)[0]
            lengths.append(orig_len)

        elif blk_type == _BLK_SPB:
            # orig_len (4 bayt)
            spb_bytes = f.read(4)
            if len(spb_bytes) < 4:
                break
            orig_len = struct.unpack_from(endian + "I", spb_bytes)[0]
            lengths.append(orig_len)

        # Blok sonuna atla (IDB, bilinmeyen bloklar ve yukaridaki bloklar icin gecerli)
        f.seek(blk_start + blk_total_len)

    return lengths


def read_packet_lengths(pcap_path: str, max_packets: int = MAX_PACKETS) -> np.ndarray:
    """PCAP veya PCAPNG dosyasindan paket uzunluklarini (orig_len) hizlica oku.

    Yalnizca baslik alanlarini okur; payload'u parse etmez.
    """
    lengths = []
    try:
        with open(pcap_path, "rb") as f:
            first4 = f.read(4)
            if len(first4) < 4:
                print(f"  HATA: Cok kisa dosya: {pcap_path}")
                return np.array([], dtype=np.int64)

            magic = struct.unpack_from("<I", first4)[0]

            if magic in (_PCAP_MAGIC_LE, _PCAP_MAGIC_LE_NS):
                f.seek(24)
                lengths = _read_pcap(f, max_packets, "<")
            elif magic in (_PCAP_MAGIC_BE, _PCAP_MAGIC_BE_NS):
                f.seek(24)
                lengths = _read_pcap(f, max_packets, ">")
            elif magic == _BLK_SHB:
                f.seek(0)
                lengths = _read_pcapng(f, max_packets)
            else:
                print(f"  HATA: Bilinmeyen format magic 0x{magic:08X}: {pcap_path}")
                return np.array([], dtype=np.int64)

    except OSError as exc:
        print(f"  HATA okunurken {pcap_path}: {exc}")

    return np.array(lengths, dtype=np.int64)


# ---------------------------------------------------------------------------
# Istatistik hesaplama
# ---------------------------------------------------------------------------
def compute_stats(lengths: np.ndarray) -> dict:
    """Tablo 1 istatistiklerini hesapla."""
    if len(lengths) == 0:
        raise ValueError("Hicbir paket okunamadi; istatistik hesaplanamaz.")
    mode_result = scipy_stats.mode(lengths, keepdims=True)
    mode_val = int(mode_result.mode[0]) if not np.isnan(mode_result.mode[0]) else int(scipy_stats.mode(lengths).mode)
    return {
        "Mode":          mode_val,
        "Median":        float(np.median(lengths)),
        "80_Percentile": float(np.percentile(lengths, 80)),
        "99_Percentile": float(np.percentile(lengths, 99)),
        "Min":           int(np.min(lengths)),
        "Max":           int(np.max(lengths)),
        "Count":         len(lengths),
    }


def paper_image_size(pstat: float) -> int:
    """Formula (2): Simage = ceil(e^(ln(Pstat) / 2))"""
    if pstat <= 0:
        return 0
    return math.ceil(math.exp(math.log(pstat) / 2))


# ---------------------------------------------------------------------------
# Yazdirma / raporlama
# ---------------------------------------------------------------------------
def print_stats_table(name: str, s: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({s['Count']:,} paket)")
    print(f"{'='*60}")
    rows = [
        ("Mode",           s["Mode"]),
        ("Median",         s["Median"]),
        ("80. Persentil",  s["80_Percentile"]),
        ("99. Persentil",  s["99_Percentile"]),
        ("Min",            s["Min"]),
        ("Max",            s["Max"]),
    ]
    print(f"  {'Istatistik':<22} {'Deger':>10}   {'Img Boyutu (Simage)':>20}")
    print(f"  {'-'*56}")
    for label, val in rows:
        simg = paper_image_size(val)
        print(f"  {label:<22} {val:>10.1f}   {simg:>10}x{simg:<10}")


def print_image_size_table(normal_s: dict, attack_s: dict) -> None:
    """Tablo 4 benzeri: her metrik icin goruntu boyutunu goster."""
    metrics = [
        ("Median (and mode)", "Median"),
        ("99. Persentil",     "99_Percentile"),
        ("Maximum size",      "Max"),
    ]
    print(f"\n{'='*60}")
    print("  Goruntu Boyutu Ozeti  (Tablo 4 benzeri)")
    print(f"{'='*60}")
    print(f"  {'Metrik':<25} {'Normal Img':>12} {'Saldiri Img':>14}")
    print(f"  {'-'*55}")
    for label, key in metrics:
        n_sz = paper_image_size(normal_s[key])
        a_sz = paper_image_size(attack_s[key])
        print(f"  {label:<25} {n_sz:>5}x{n_sz:<6}  {a_sz:>5}x{a_sz}")


# ---------------------------------------------------------------------------
# Grafik
# ---------------------------------------------------------------------------
def plot_distributions(
    normal_lengths: np.ndarray,
    attack_lengths: np.ndarray,
    normal_stats: dict,
    attack_stats: dict,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Paket Boyutu Dagilimi – Normal vs Saldiri", fontsize=14, fontweight="bold")

    clip_max = max(normal_stats["99_Percentile"], attack_stats["99_Percentile"]) * 1.05
    bins = np.linspace(0, clip_max, 80)

    colors = {"Normal": "#2196F3", "Saldiri": "#F44336"}

    # --- Sol panel: histogram ---
    ax = axes[0]
    n_clip = normal_lengths[normal_lengths <= clip_max]
    a_clip = attack_lengths[attack_lengths <= clip_max]
    ax.hist(n_clip, bins=bins, alpha=0.65, color=colors["Normal"],
            label=f"Normal (n={len(normal_lengths):,})", density=True)
    ax.hist(a_clip, bins=bins, alpha=0.65, color=colors["Saldiri"],
            label=f"Saldiri (n={len(attack_lengths):,})", density=True)

    for label, s, c in [
        ("Normal", normal_stats, colors["Normal"]),
        ("Saldiri", attack_stats, colors["Saldiri"]),
    ]:
        ax.axvline(s["Median"], color=c, linestyle="--", linewidth=1.4,
                   label=f"{label} Median={s['Median']:.0f}")
        ax.axvline(s["99_Percentile"], color=c, linestyle=":", linewidth=1.4,
                   label=f"{label} 99p={s['99_Percentile']:.0f}")

    ax.set_xlabel("Paket Uzunlugu (bayt)", fontsize=11)
    ax.set_ylabel("Yogunluk", fontsize=11)
    ax.set_title("Dagilim (0 – 99. persentil)", fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- Sag panel: kutu grafigi ---
    ax2 = axes[1]
    data_to_plot = [
        normal_lengths[normal_lengths <= clip_max],
        attack_lengths[attack_lengths <= clip_max],
    ]
    bp = ax2.boxplot(
        data_to_plot,
        tick_labels=["Normal", "Saldiri"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], [colors["Normal"], colors["Saldiri"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel("Paket Uzunlugu (bayt)", fontsize=11)
    ax2.set_title("Kutu Grafigi (0 – 99. persentil)", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Istatistik notu (sag alt)
    note_lines = []
    for label, s in [("Normal", normal_stats), ("Saldiri", attack_stats)]:
        note_lines.append(
            f"{label}: mode={s['Mode']}  med={s['Median']:.0f}  "
            f"80p={s['80_Percentile']:.0f}  99p={s['99_Percentile']:.0f}  "
            f"min={s['Min']}  max={s['Max']}"
        )
    fig.text(
        0.5, -0.02, "\n".join(note_lines),
        ha="center", fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrafik kaydedildi: {out_path}")


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Normal ---
    print(f"Normal PCAP okunuyor: {NORMAL_PCAP}")
    if not os.path.isfile(NORMAL_PCAP):
        raise FileNotFoundError(f"Dosya bulunamadi: {NORMAL_PCAP}")
    normal_lengths = read_packet_lengths(NORMAL_PCAP)
    print(f"  {len(normal_lengths):,} paket okundu.")

    # --- Saldiri ---
    attack_files = sorted(glob.glob(os.path.join(MALICIOUS_DIR, "*.pcap")))
    if not attack_files:
        raise FileNotFoundError(f"Saldiri PCAP bulunamadi: {MALICIOUS_DIR}/*.pcap")

    print(f"\nSaldiri PCAP dosyalari ({len(attack_files)} adet):")
    all_attack_lengths = []
    for f in attack_files:
        lens = read_packet_lengths(f)
        print(f"  {os.path.basename(f)}: {len(lens):,} paket")
        all_attack_lengths.append(lens)
    attack_lengths = np.concatenate(all_attack_lengths)
    print(f"  Toplam: {len(attack_lengths):,} paket")

    # --- Istatistikler ---
    normal_stats = compute_stats(normal_lengths)
    attack_stats = compute_stats(attack_lengths)

    print_stats_table("NORMAL TRAFIK", normal_stats)
    print_stats_table("SALDIRI TRAFIGI", attack_stats)
    print_image_size_table(normal_stats, attack_stats)

    # --- Grafik ---
    plot_distributions(normal_lengths, attack_lengths, normal_stats, attack_stats, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
