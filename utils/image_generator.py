"""
CSV -> 6x6 Grayscale Image Generator
====================================

Reads a preprocessed CSV (every feature already in the [0, 255] range) and
turns each row into a 6x6 grayscale PNG using PIL (no matplotlib in the hot
loop -> hundreds of times faster than the old implementation).

Split policy:
    * 80% of the NORMAL rows  -> train set
    * 20% of the NORMAL rows  -> test set
    * 100% of the ATTACK rows -> test set

Output layout:
    <img_output_path>/
        train/normal/sample_XXXXX_normal.png
        test/normal/sample_XXXXX_normal.png
        test/attack/sample_XXXXX_attack.png
        example_visualization.png
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


IMG_SIZE = 6
TARGET_FEATURES = IMG_SIZE * IMG_SIZE  # 36
NORMAL_LABEL = 'normal'
ATTACK_LABEL = 'attack'


def _adjust_row_length(row, target_len=TARGET_FEATURES):
    """Pad with zeros or truncate so that `row` has exactly `target_len` values."""
    if len(row) == target_len:
        return row
    if len(row) < target_len:
        return np.append(row, np.zeros(target_len - len(row)))
    return row[:target_len]


def _row_to_image(row, upscale=1):
    """Return a PIL Image for one feature row (clipped to [0, 255], uint8)."""
    adjusted = _adjust_row_length(row)
    pixels = np.clip(adjusted, 0, 255).astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE)
    img = Image.fromarray(pixels, mode='L')
    if upscale and upscale > 1:
        img = img.resize(
            (IMG_SIZE * upscale, IMG_SIZE * upscale),
            resample=Image.NEAREST,
        )
    return img


def _save_row_as_image(row, filepath, upscale=1):
    """Render a single feature row as a grayscale PNG via PIL."""
    _row_to_image(row, upscale=upscale).save(filepath, format='PNG', optimize=True)


def _detect_label_column(df, preferred=('Attack Category', 'class', 'classification.', 'Label')):
    """Return the most likely label/class column name present in `df`."""
    for name in preferred:
        if name in df.columns:
            return name
    return df.columns[-1]


def _save_split(features, split_name, class_name, output_root,
                start_idx=0, upscale=1):
    """Save every row under <output_root>/<split>/<class>/sample_XXXXX_<class>.png."""
    out_dir = os.path.join(output_root, split_name, class_name)
    os.makedirs(out_dir, exist_ok=True)

    total = len(features)
    for i, row in enumerate(features):
        filename = os.path.join(
            out_dir, f"sample_{start_idx + i:05d}_{class_name}.png"
        )
        _save_row_as_image(row, filename, upscale=upscale)

        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"    [{split_name}/{class_name}] {i + 1}/{total}")

    return total


def csv_to_6x6_images(csv_input_path, img_output_path,
                      test_size=0.20, random_state=42, upscale=1):
    """
    Convert each row of the CSV into a 6x6 grayscale image, splitting the data
    so that 80% of the normal rows go to the train set, 20% of the normals plus
    all attack rows go to the test set.

    Parameters
    ----------
    csv_input_path : str
        Path to the preprocessed CSV file.
    img_output_path : str
        Root directory where the `train/` and `test/` folders will be created.
    test_size : float, default 0.20
        Fraction of the NORMAL samples that go into the test split.
    random_state : int, default 42
        Seed for the normal-samples split (reproducibility).
    upscale : int, default 1
        If > 1, every 6x6 image is nearest-neighbor upscaled to
        (6 * upscale) x (6 * upscale). Useful only for human inspection;
        training with CNNs works fine on the native 6x6.
    """
    df = pd.read_csv(csv_input_path)
    label_col = _detect_label_column(df)

    feature_columns = [c for c in df.columns if c != label_col]
    features_df = df[feature_columns]
    labels = df[label_col].astype(str).str.strip()

    print(f"CSV file: {csv_input_path}")
    print(f"Label column: '{label_col}'")
    print(f"Total number of features: {len(feature_columns)}")
    print(f"Total number of data rows: {len(df)}")
    print(f"Class distribution:")
    for cls, cnt in labels.value_counts().items():
        print(f"  {cls}: {cnt}")

    if len(feature_columns) != TARGET_FEATURES:
        print(f"Warning: feature count is {len(feature_columns)}, "
              f"expected {TARGET_FEATURES}. "
              f"{'Padding with zeros.' if len(feature_columns) < TARGET_FEATURES else 'Truncating extras.'}")

    features_array = features_df.values
    labels_array = labels.values

    normal_mask = labels_array == NORMAL_LABEL
    attack_mask = ~normal_mask
    normal_features = features_array[normal_mask]
    attack_features = features_array[attack_mask]

    print(f"\nSplitting normals: {(1 - test_size) * 100:.0f}% train / "
          f"{test_size * 100:.0f}% test  |  All attacks -> test")
    print(f"  Normals: {len(normal_features)}  |  Attacks: {len(attack_features)}")

    if len(normal_features) == 0:
        raise ValueError(f"No rows with label '{NORMAL_LABEL}' found in '{label_col}'.")

    normal_train, normal_test = train_test_split(
        normal_features, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"  -> train/normal: {len(normal_train)}")
    print(f"  -> test/normal : {len(normal_test)}")
    print(f"  -> test/attack : {len(attack_features)}")

    os.makedirs(img_output_path, exist_ok=True)

    print(f"\nGenerating images via PIL (upscale={upscale})...")
    _save_split(normal_train, 'train', NORMAL_LABEL, img_output_path,
                start_idx=0, upscale=upscale)
    _save_split(normal_test, 'test', NORMAL_LABEL, img_output_path,
                start_idx=0, upscale=upscale)
    _save_split(attack_features, 'test', ATTACK_LABEL, img_output_path,
                start_idx=0, upscale=upscale)

    print(f"\nAll images saved under: {img_output_path}/")

    _save_example_visualization(features_array[0], img_output_path, upscale=upscale)


def _save_example_visualization(first_row, img_output_path, upscale=1):
    """Save a side-by-side preview of the first sample (image + histogram).

    This is the only place matplotlib is used, imported lazily so the main
    pipeline has zero matplotlib overhead.
    """
    import matplotlib.pyplot as plt

    adjusted = _adjust_row_length(first_row)
    preview_img = np.asarray(_row_to_image(first_row, upscale=max(upscale, 32)))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(preview_img, cmap='gray', vmin=0, vmax=255)
    plt.title(f'{IMG_SIZE}x{IMG_SIZE} grayscale (first sample)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.hist(adjusted, bins=20, alpha=0.7)
    plt.title('Feature value distribution')
    plt.xlabel('Feature value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    out_path = os.path.join(img_output_path, 'example_visualization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Example visualization saved: {out_path}")


if __name__ == "__main__":
    csv_input_path = 'preprocessed_data/onehot_preprocessed_wustlehms_modified_columns.csv'
    img_output_path = 'wustlehms_images_onehot'

    csv_to_6x6_images(
        csv_input_path,
        img_output_path,
        test_size=0.20,
        random_state=42,
        upscale=1,        # set to 8 or 16 for human-readable previews
    )

    print("\n=== Conversion Complete ===")
    print("  train/normal : 80% of normal rows")
    print("  test/normal  : 20% of normal rows")
    print("  test/attack  : 100% of attack rows")
    print(f"Each row is a {IMG_SIZE}x{IMG_SIZE} grayscale PNG (features fixed to [0, 255]).")
