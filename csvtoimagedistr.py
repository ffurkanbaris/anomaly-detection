import os
import argparse
import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from PIL import Image

# ================== YARDIMCI FONKSİYONLAR ==================

def load_multiple_csvs(input_path: str, label_col: str, normal_label_value) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.isdir(input_path):
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    elif os.path.isfile(input_path):
        csv_files = [input_path]
    else:
        raise FileNotFoundError(f"Yol bulunamadı: {input_path}")

    all_normal = []
    all_attack = []
    target_col_clean = label_col.strip().lower()

    for path in csv_files:
        print(f"İşleniyor: {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if target_col_clean not in df.columns:
            print(f"!!! '{label_col}' bulunamadı. Atlanıyor.")
            continue

        # Sayısal görünümlü ama aslında karmaşık string olan sütunları temizle
        # (Hata veren uzun hash/string yapılarını tespit eder)
        for col in df.columns:
            if col != target_col_clean and df[col].dtype == 'object':
                # İlk 100 satırda numerik olmayan uzun stringler var mı bak
                sample = df[col].dropna().head(100).astype(str)
                if sample.str.len().mean() > 20: # Hash veya uzun ID tespiti
                    df = df.drop(columns=[col])

        mask_normal = (df[target_col_clean].astype(str) == str(normal_label_value))
        X_normal = df[mask_normal].drop(columns=[target_col_clean])
        X_attack = df[~mask_normal].drop(columns=[target_col_clean])

        if not X_normal.empty: all_normal.append(X_normal)
        if not X_attack.empty: all_attack.append(X_attack)

    if not all_normal:
        raise ValueError(f"HATA: Hiçbir dosyada uygun veri bulunamadı.")

    return pd.concat(all_normal, ignore_index=True), (pd.concat(all_attack, ignore_index=True) if all_attack else pd.DataFrame())

def _to_numeric_safe(X):
    """Convert columns to numeric; non-numeric (e.g. hash strings) become NaN for imputer."""
    if hasattr(X, "iloc"):
        return X.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").values


def build_preprocess_pipeline(X: pd.DataFrame):
    # Sayısal ve kategorik sütunları ayır
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipeline = Pipeline([
            ("to_numeric", FunctionTransformer(_to_numeric_safe)),
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    col_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    return Pipeline([("col_tf", col_transformer), ("qt", qt)])

# PCA ve Görüntü Fonksiyonları (Öncekiyle Aynı)
def apply_pca(X_norm: np.ndarray, n_components: int = 32):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    return X_pca, pca

def scale_for_image_with_reference(X_ref: np.ndarray, X_target: np.ndarray):
    mins, maxs = X_ref.min(axis=0), X_ref.max(axis=0)
    ranges = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    def normalize(data):
        if data.size == 0: return data
        return ((data - mins) / ranges * 255).clip(0, 255).astype(np.uint8)
    return normalize(X_ref), normalize(X_target)

def _window_to_image(window_data: np.ndarray, h: int, w: int, c: int):
    if window_data.shape[0] != h: return None
    img_array = window_data.reshape(h, w)
    mode = "L" if c == 1 else "RGB"
    if c == 3: img_array = np.stack([img_array]*3, axis=-1)
    return img_array, mode

def main(args):
    if args.img_width != args.n_components:
        args.img_width = args.n_components

    X_normal, X_attack = load_multiple_csvs(args.input_path, args.label_col, args.normal_label)

    # Pipeline kurulumu
    pipe = build_preprocess_pipeline(X_normal)
    X_normal_norm = pipe.fit_transform(X_normal)
    X_normal_pca, pca_model = apply_pca(X_normal_norm, n_components=args.n_components)
    
    X_attack_scaled = np.array([])
    if not X_attack.empty:
        # Transform sırasında numerik olmayan yeni değer gelirse hata vermemesi için
        X_attack_norm = pipe.transform(X_attack)
        X_attack_pca = pca_model.transform(X_attack_norm)
        X_normal_scaled, X_attack_scaled = scale_for_image_with_reference(X_normal_pca, X_attack_pca)
    else:
        X_normal_scaled, _ = scale_for_image_with_reference(X_normal_pca, np.array([]))

    # Klasörleri oluştur ve Kaydet
    for d in [args.output_train_dir, args.output_test_normal_dir, args.output_test_attack_dir]:
        os.makedirs(d, exist_ok=True)

    start_indices = list(range(0, len(X_normal_scaled) - args.window_size + 1, args.stride))
    np.random.RandomState(args.random_state).shuffle(start_indices)
    split = int(len(start_indices) * args.train_ratio)
    
    print("Görüntüler kaydediliyor...")
    # Kayıt döngüleri
    for i, idx in enumerate(start_indices[:split]):
        win = X_normal_scaled[idx : idx + args.window_size, :]
        res = _window_to_image(win, args.img_height, args.img_width, args.img_channels)
        Image.fromarray(res[0], mode=res[1]).save(Path(args.output_train_dir) / f"n_tr_{i:06d}.png")

    for i, idx in enumerate(start_indices[split:]):
        win = X_normal_scaled[idx : idx + args.window_size, :]
        res = _window_to_image(win, args.img_height, args.img_width, args.img_channels)
        Image.fromarray(res[0], mode=res[1]).save(Path(args.output_test_normal_dir) / f"n_ts_{i:06d}.png")

    attack_count = 0
    if X_attack_scaled.size > 0:
        for i in range(0, len(X_attack_scaled) - args.window_size + 1, args.stride):
            win = X_attack_scaled[i : i + args.window_size, :]
            res = _window_to_image(win, args.img_height, args.img_width, args.img_channels)
            Image.fromarray(res[0], mode=res[1]).save(Path(args.output_test_attack_dir) / f"atk_{i:06d}.png")
            attack_count += 1

    print(f"Tamamlandı! Normal(Train): {split}, Normal(Test): {len(start_indices)-split}, Attack: {attack_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--normal-label", type=int, default=0)
    parser.add_argument("--output-train-dir", type=str, default="IOT/train")
    parser.add_argument("--output-test-normal-dir", type=str, default="IOT/test/normal")
    parser.add_argument("--output-test-attack-dir", type=str, default="IOT/test/attack")
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=32)
    parser.add_argument("--img-channels", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    main(parser.parse_args())