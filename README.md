# Data to Image & DROCC Anomaly Detection

This project is a machine learning system that converts network packets (PCAP files, CSV data) into images and performs anomaly detection using the DROCC (Deep Robust One-Class Classification) algorithm.

## Features

- Converting PCAP files to grayscale images
- One-class learning with LeNet-5 based DROCC model
- Adversarial training for anomaly detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Converting PCAP Files to Images

```bash
python pcap_to_image.py
```

This script converts PCAP files to PNG images and saves them to the `pcap_images/` folder.

### 2. Model Training

```bash
python train_drocc_packets.py
```

The model is trained only on normal packets and is used for anomaly detection.

## Folder Structure

```
.
├── pcap_to_image.py          # PCAP -> Image converter
├── train_drocc_packets.py    # DROCC model training
├── requirements.txt          # Python dependencies
├── pcap_images/              # Generated images
│   ├── Normal/              # Normal packet images
│   └── Attack/              # Attack packet images (for testing)
└── IP-Based/                # PCAP files
    ├── Normal/
    └── Malicious/
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Scapy
- Pillow (PIL)
- torchvision

## License

MIT
