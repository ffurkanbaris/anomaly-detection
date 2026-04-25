"""
WUSTL-EHMS Dataset Preprocessing with One-Hot Encoding
======================================================

Adapted from the NSL-KDD version to work on the WUSTL-EHMS dataset
(`./datasets/wustlehms.csv`).

Processing Steps:
1. Feature Selection  - Drop identifier / constant columns (CONFIG['drop_features'])
2. Categorical Encoding - One-hot encoding for CONFIG['categorical_features']
                         (dummy columns are scaled from 0/1 to 0/255)
3. Data Transformation - log10(x + 1) scaling for heavy-tailed numeric columns
4. Value Mapping       - Binary / discrete / rate feature mapping
5. Normalization       - Remaining numerics scaled to [0, 255] via MinMax
6. Binary Classification - Collapse CONFIG['attack_classes'] into 'attack'
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

warnings.filterwarnings('ignore')

# Configuration tailored for the WUSTL-EHMS dataset (./datasets/wustlehms.csv).
# Mirrors the one used by data_preprocessing_label_encode.py so the two
# pipelines stay in sync.
CONFIG = {
    'input_file': './datasets/wustlehms.csv',
    'output_dir': './preprocessed_data',
    'random_state': 42,

    # Target / classification
    'target_column': 'Attack Category',
    'binary_label_column': 'Label',
    'normal_class': 'normal',
    'attack_classes': ['Spoofing', 'Data Alteration'],

    # Identifier + constant + redundant label columns to drop entirely.
    # The four "numeric constants" below also carry zero information in
    # WUSTL-EHMS (verified via df[col].nunique() == 1) and dropping them
    # brings the final feature count to exactly 36 = 6*6 pixels after
    # one-hot encoding of 'Flgs'.
    'drop_features': [
        'Dir', 'Trans',
        'SrcAddr', 'DstAddr', 'SrcMac', 'DstMac',
        'Sport', 'Dport',
        'Packet_num',
        'Label',
        'SrcGap', 'DstGap', 'DIntPktAct', 'dMinPktSz',
    ],

    'categorical_features': ['Flgs'],

    'log_features': [
        'SrcBytes', 'DstBytes', 'TotBytes',
        'SrcLoad', 'DstLoad', 'Load', 'Rate',
        'SrcJitter', 'DstJitter',
        'SIntPkt', 'DIntPkt', 'SIntPktAct',
        'Dur', 'TotPkts', 'Loss',
        'sMaxPktSz', 'dMaxPktSz', 'sMinPktSz',
    ],

    # WUSTL-EHMS pLoss columns are percentages up to ~33, NOT ratios in [0, 1],
    # so they're handled by the generic MinMax normalizer instead of *255.
    'rate_features': [],

    'binary_features': [],
    'discrete_features': [],

    'biometric_features': ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA',
                           'Heart_rate', 'Resp_Rate', 'ST'],
}

# Legacy module-level aliases used by helper functions below.
CATEGORICAL_FEATURES = CONFIG['categorical_features']
LOW_INFORMATION_FEATURES = CONFIG['drop_features']
LOG_SCALING_FEATURES = CONFIG['log_features']
BINARY_FEATURES = CONFIG['binary_features']
DISCRETE_FEATURES = CONFIG['discrete_features']
RATE_FEATURES = CONFIG['rate_features']


def load_dataset(file_path):
    """Load the WUSTL-EHMS CSV and report a summary."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✓ Dataset loaded successfully")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {len(df.columns)}")

    target_column = CONFIG.get('target_column')
    if target_column and target_column in df.columns:
        print(f"  - Class distribution ({target_column}):")
        for class_name, count in df[target_column].value_counts().items():
            print(f"    {class_name}: {count}")

    return df


def remove_low_information_features(df, features_to_remove):
    """Drop the columns listed in `features_to_remove` (if they exist)."""
    print(f"Removing {len(features_to_remove)} low information gain features...")

    existing_features = [f for f in features_to_remove if f in df.columns]
    missing_features = [f for f in features_to_remove if f not in df.columns]

    if missing_features:
        print(f"  Warning: Features not found in dataset: {missing_features}")

    if existing_features:
        df_reduced = df.drop(columns=existing_features)
        print(f"  ✓ Removed features: {existing_features}")
        print(f"  - Remaining features: {len(df_reduced.columns)}")
    else:
        df_reduced = df.copy()
        print(f"  No features were removed")

    return df_reduced


def encode_categorical_features(df, categorical_features, dummy_on_value=255):
    """
    One-hot encode the given categorical columns.

    Dummy columns use `dummy_on_value` for the "present" class so the output is
    directly in the 8-bit pixel range ({0, 255} instead of {0, 1}).
    """
    print("Applying one-hot encoding to categorical features...")

    existing_categorical = [f for f in categorical_features if f in df.columns]
    missing_categorical = [f for f in categorical_features if f not in df.columns]

    if missing_categorical:
        print(f"  Warning: Categorical features not found: {missing_categorical}")

    if not existing_categorical:
        print("  No categorical features found to encode")
        return df

    df_work = df.copy()

    # Strip whitespace on string-like columns (WUSTL-EHMS 'Flgs' has padding).
    for feature in existing_categorical:
        if not pd.api.types.is_numeric_dtype(df_work[feature]):
            df_work[feature] = df_work[feature].astype(str).str.strip()

    categorical_data = df_work[existing_categorical].copy()

    dummy_columns = []
    for feature in existing_categorical:
        unique_values = sorted(df_work[feature].unique())
        feature_columns = [f"{feature}_{value}" for value in unique_values]
        dummy_columns.extend(feature_columns)

    print(f"  - Features to encode: {existing_categorical}")
    print(f"  - Total dummy columns to create: {len(dummy_columns)}")

    # Label encode first (so OneHotEncoder sees a consistent integer domain).
    categorical_encoded = categorical_data.apply(LabelEncoder().fit_transform)

    encoder = OneHotEncoder(sparse_output=False)
    categorical_onehot = encoder.fit_transform(categorical_encoded)

    categorical_df = pd.DataFrame(
        categorical_onehot,
        columns=dummy_columns,
        index=df_work.index,
    )

    # Scale dummies from 0/1 to 0/dummy_on_value for pixel compatibility.
    if dummy_on_value != 1:
        categorical_df = categorical_df * dummy_on_value

    df_encoded = df_work.drop(columns=existing_categorical).join(categorical_df)

    print(f"  ✓ One-hot encoding completed (on-value: {dummy_on_value})")
    print(f"  - Final shape: {df_encoded.shape}")

    return df_encoded


def apply_log_scaling(df, log_features):
    """log10(x + 1) on the given columns."""
    print("Applying log scaling to numerical features...")

    df_scaled = df.copy()
    existing_log_features = [f for f in log_features if f in df.columns]
    missing_log_features = [f for f in log_features if f not in df.columns]

    if missing_log_features:
        print(f"  Warning: Log scaling features not found: {missing_log_features}")

    if existing_log_features:
        for feature in existing_log_features:
            original_min = df_scaled[feature].min()
            original_max = df_scaled[feature].max()
            df_scaled[feature] = np.log10(df_scaled[feature] + 1)
            new_min = df_scaled[feature].min()
            new_max = df_scaled[feature].max()
            print(f"    {feature}: [{original_min:.2f}, {original_max:.2f}] -> "
                  f"[{new_min:.4f}, {new_max:.4f}]")
        print(f"  ✓ Log scaling applied to {len(existing_log_features)} features")
    else:
        print("  No features found for log scaling")

    return df_scaled


def apply_binary_mapping(df, binary_features):
    """0/1 -> 0/255 on the given columns."""
    print("Applying binary mapping (0->0, 1->255)...")

    df_mapped = df.copy()
    existing_binary = [f for f in binary_features if f in df.columns]
    missing_binary = [f for f in binary_features if f not in df.columns]

    if missing_binary:
        print(f"  Warning: Binary features not found: {missing_binary}")

    if existing_binary:
        for feature in existing_binary:
            df_mapped.loc[df_mapped[feature] == 1, feature] = 255
        print(f"  ✓ Binary mapping applied to: {existing_binary}")
        for feature in existing_binary:
            print(f"    {feature} values: {df_mapped[feature].value_counts().to_dict()}")
    else:
        print("  No binary features found for mapping")

    return df_mapped


def apply_discrete_mapping(df, discrete_features):
    """0/1/2 -> 85/170/255 on the given columns."""
    print("Applying discrete mapping (0->85, 1->170, 2->255)...")

    df_mapped = df.copy()
    existing_discrete = [f for f in discrete_features if f in df.columns]
    missing_discrete = [f for f in discrete_features if f not in df.columns]

    if missing_discrete:
        print(f"  Warning: Discrete features not found: {missing_discrete}")

    if existing_discrete:
        for feature in existing_discrete:
            df_mapped.loc[df_mapped[feature] == 0, feature] = 85
            df_mapped.loc[df_mapped[feature] == 1, feature] = 170
            df_mapped.loc[df_mapped[feature] == 2, feature] = 255
        print(f"  ✓ Discrete mapping applied to: {existing_discrete}")
        for feature in existing_discrete:
            print(f"    {feature} values: {df_mapped[feature].value_counts().to_dict()}")
    else:
        print("  No discrete features found for mapping")

    return df_mapped


def apply_rate_scaling(df, rate_features):
    """Multiply [0, 1] rates by 255."""
    print("Applying rate scaling (multiply by 255)...")

    df_scaled = df.copy()
    existing_rate = [f for f in rate_features if f in df.columns]
    missing_rate = [f for f in rate_features if f not in df.columns]

    if missing_rate:
        print(f"  Warning: Rate features not found: {missing_rate}")

    if existing_rate:
        for feature in existing_rate:
            df_scaled[feature] = df_scaled[feature] * 255
        print(f"  ✓ Rate scaling applied to: {existing_rate}")
    else:
        print("  No rate features found for scaling")

    return df_scaled


def normalize_remaining_features(df, exclude_features=None):
    """MinMax-scale every unprocessed numeric column to [0, 255]."""
    if exclude_features is None:
        exclude_features = []

    print("Normalizing remaining numerical features to [0, 255] range...")

    df_normalized = df.copy()
    target_column = CONFIG.get('target_column')

    # Columns already handled by earlier steps, plus the target itself.
    processed_features = set(LOG_SCALING_FEATURES + BINARY_FEATURES +
                             DISCRETE_FEATURES + RATE_FEATURES +
                             list(exclude_features))
    if target_column:
        processed_features.add(target_column)

    # One-hot dummy columns start with "<feature>_" and are already 0/255.
    onehot_prefixes = tuple(f"{c}_" for c in CATEGORICAL_FEATURES)

    remaining_features = []
    for col in df_normalized.columns:
        if col in processed_features:
            continue
        if col.startswith(onehot_prefixes):
            continue
        if df_normalized[col].dtype in ['int64', 'float64']:
            remaining_features.append(col)

    if remaining_features:
        print(f"  Features to normalize: {remaining_features}")
        for feature in remaining_features:
            original_range = (df_normalized[feature].min(), df_normalized[feature].max())

            if original_range[1] > 1000:
                df_normalized[feature] = np.log10(df_normalized[feature] + 1)

            scaler = MinMaxScaler(feature_range=(0, 255))
            df_normalized[feature] = scaler.fit_transform(df_normalized[[feature]]).flatten()

            new_range = (df_normalized[feature].min(), df_normalized[feature].max())
            print(f"    {feature}: [{original_range[0]:.2f}, {original_range[1]:.2f}] -> "
                  f"[{new_range[0]:.2f}, {new_range[1]:.2f}]")
        print(f"  ✓ Normalization applied to {len(remaining_features)} features")
    else:
        print("  No remaining features found for normalization")

    return df_normalized


def convert_to_binary_classification(df, target_column=None, attack_classes=None,
                                     normal_class=None):
    """Collapse every label in `attack_classes` into the string 'attack'."""
    print("Converting to binary classification...")

    target_column = target_column or CONFIG.get('target_column')
    attack_classes = attack_classes if attack_classes is not None \
        else CONFIG.get('attack_classes', [])
    normal_class = normal_class or CONFIG.get('normal_class', 'normal')

    if target_column not in df.columns:
        print(f"  Warning: Target column '{target_column}' not found")
        return df

    df_binary = df.copy()

    print(f"  Original class distribution:")
    for class_name, count in df_binary[target_column].value_counts().items():
        print(f"    {class_name}: {count}")

    for attack_type in attack_classes:
        df_binary.loc[df_binary[target_column] == attack_type, target_column] = 'attack'

    if normal_class != 'normal':
        df_binary.loc[df_binary[target_column] == normal_class, target_column] = 'normal'

    print(f"  ✓ Binary classification applied")
    print(f"  New class distribution:")
    for class_name, count in df_binary[target_column].value_counts().items():
        print(f"    {class_name}: {count}")

    return df_binary


def convert_data_types(df):
    """Round and downcast clean float columns to int64 (pixel-like)."""
    print("Converting data types...")

    df_converted = df.copy()
    target_column = CONFIG.get('target_column')
    float_columns = []

    for column in df_converted.columns:
        if df_converted[column].dtype == 'float64' and column != target_column:
            if df_converted[column].notna().all():
                df_converted[column] = np.round(df_converted[column]).astype('int64')
                float_columns.append(column)

    if float_columns:
        print(f"  ✓ Converted {len(float_columns)} columns from float64 to int64")
    else:
        print("  No columns needed conversion")

    return df_converted


def reorder_columns(df, target_column=None):
    """Place the target first, then one-hot columns, then everything else."""
    print("Reordering columns...")

    target_column = target_column or CONFIG.get('target_column')
    if target_column not in df.columns:
        print(f"  Warning: Target column '{target_column}' not found")
        return df

    onehot_features = []
    original_features = []

    for col in df.columns:
        if col == target_column:
            continue
        if any(col.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES):
            onehot_features.append(col)
        else:
            original_features.append(col)

    new_order = [target_column] + onehot_features + original_features
    df_reordered = df[new_order]

    print(f"  ✓ Columns reordered")
    print(f"    - Target column: 1")
    print(f"    - One-hot encoded features: {len(onehot_features)}")
    print(f"    - Original features: {len(original_features)}")

    return df_reordered


def save_preprocessed_data(df, output_dir, filename_prefix="onehot_preprocessed",
                           dataset_tag="wustlehms"):
    """Save the preprocessed dataframe plus normal/attack splits and a reordered copy."""
    print("Saving preprocessed data...")

    os.makedirs(output_dir, exist_ok=True)
    target_column = CONFIG.get('target_column')

    complete_file = os.path.join(output_dir, f"{filename_prefix}_{dataset_tag}.csv")
    df.to_csv(complete_file, index=False)
    print(f"  ✓ Complete dataset saved: {complete_file}")

    modified_file = os.path.join(
        output_dir, f"{filename_prefix}_{dataset_tag}_modified_columns.csv")
    df_reordered = reorder_columns(df)
    df_reordered.to_csv(modified_file, index=False)
    print(f"  ✓ Modified columns dataset saved: {modified_file}")

    if target_column and target_column in df.columns:
        normal_df = df[df[target_column] == 'normal']
        attack_df = df[df[target_column] == 'attack']

        normal_file = os.path.join(output_dir, f"{filename_prefix}_{dataset_tag}_normal.csv")
        attack_file = os.path.join(output_dir, f"{filename_prefix}_{dataset_tag}_attack.csv")

        normal_df.to_csv(normal_file, index=False)
        attack_df.to_csv(attack_file, index=False)

        print(f"  ✓ Normal data saved: {normal_file} ({len(normal_df)} samples)")
        print(f"  ✓ Attack data saved: {attack_file} ({len(attack_df)} samples)")

    feature_file = os.path.join(
        output_dir, f"{filename_prefix}_{dataset_tag}_feature_list.txt")
    feature_columns = [col for col in df.columns if col != target_column]
    with open(feature_file, 'w') as f:
        f.write("Feature List (excluding target column):\n")
        f.write("=" * 50 + "\n")
        for i, feature in enumerate(feature_columns, 1):
            f.write(f"{i:2d}. {feature}\n")
    print(f"  ✓ Feature list saved: {feature_file}")


def main():
    print("=" * 60)
    print("WUSTL-EHMS Dataset Preprocessing with One-Hot Encoding")
    print("=" * 60)

    try:
        df = load_dataset(CONFIG['input_file'])

        df = remove_low_information_features(df, LOW_INFORMATION_FEATURES)
        df = encode_categorical_features(df, CATEGORICAL_FEATURES, dummy_on_value=255)
        df = apply_log_scaling(df, LOG_SCALING_FEATURES)
        df = apply_binary_mapping(df, BINARY_FEATURES)
        df = apply_discrete_mapping(df, DISCRETE_FEATURES)
        df = apply_rate_scaling(df, RATE_FEATURES)
        df = normalize_remaining_features(df)
        df = convert_to_binary_classification(df)
        df = convert_data_types(df)

        save_preprocessed_data(df, CONFIG['output_dir'], "onehot_preprocessed")

        print("\n" + "=" * 60)
        print("✓ One-hot encoding preprocessing completed successfully!")
        print(f"✓ Final dataset shape: {df.shape}")
        print(f"✓ Output saved to: {CONFIG['output_dir']}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
