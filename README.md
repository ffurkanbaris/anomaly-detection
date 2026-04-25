# Data to Image & DROCC Anomaly Detection

This project converts network packets (PCAP files) into grayscale images and performs anomaly detection using the DROCC (Deep Robust One-Class Classification) algorithm.

## Features

- Converting PCAP files to grayscale images
- One-class learning with LeNet-5 based DROCC model
- Adversarial training for anomaly detection
- ROC/confusion matrix evaluation and Isolation Forest baseline

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Converting PCAP Files to Images

**Basic converter:**
```bash
python pcap_to_image.py
```

**Flow-level converter (v2):**
```bash
python pcaptoimagev2.py
```

Place PCAP files in the input directory; images are saved in the configured output folders.

### 2. Model Training

```bash
python traindrocc_v2.py
```

Optional arguments:

```bash
python traindrocc_v2.py --train_normal_dir images/train/normal --test_normal_dir images/test/normal --test_attack_dir images/test/attack --epochs 20
```

The model trains only on normal packets and detects anomalies at test time.

## Requirements

- Python 3.7+
- PyTorch
- NumPy, scikit-learn
- torchvision
- Pillow (PIL)
- scapy
- matplotlib, tqdm

## License

MIT
