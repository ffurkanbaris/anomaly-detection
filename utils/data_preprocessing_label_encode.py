"""
WUSTL-EHMS Dataset Preprocessing with Automated Label Encoding
==============================================================

This script preprocesses the WUSTL-EHMS dataset (`./datasets/wustlehms.csv`) so
that every feature lands in the 8-bit [0, 255] range, making the output CSV
directly consumable by image-based anomaly detection models.

Processing Steps:
1. Feature Selection - Drop identifier / constant columns (CONFIG['drop_features'])
2. Data Transformation - log10(x + 1) scaling for heavy-tailed numeric features
3. Categorical Encoding - Automated label encoding for CONFIG['categorical_features']
4. Value Mapping - Binary / discrete / rate feature mapping
5. Normalization - Remaining numerics scaled to [0, 255] via MinMax
6. Binary Classification - Collapse CONFIG['attack_classes'] into a single 'attack' label
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuration tailored for the WUSTL-EHMS dataset (./datasets/wustlehms.csv)
# Dataset summary:
#   - 16318 rows, 45 columns
#   - Target column: 'Attack Category' with classes {'normal', 'Spoofing', 'Data Alteration'}
#   - Binary label column 'Label' (0 = normal, 1 = attack) also available
#   - Constant (single-value) columns: 'Dir', 'Trans'
#   - Identifier / high-cardinality columns (not useful for the model): IPs, MACs, ports, packet index
#   - Only one real categorical column worth encoding: 'Flgs'
CONFIG = {
    'input_file': './datasets/wustlehms.csv',
    'output_dir': './preprocessed_data',
    'random_state': 42,

    # Target / classification
    'target_column': 'Attack Category',
    'binary_label_column': 'Label',            # already 0/1 in the raw CSV
    'normal_class': 'normal',
    'attack_classes': ['Spoofing', 'Data Alteration'],

    # Columns to drop entirely (identifiers + constants + redundant labels)
    'drop_features': [
        'Dir', 'Trans',                        # constant single-value columns
        'SrcAddr', 'DstAddr', 'SrcMac', 'DstMac',  # identifiers
        'Sport', 'Dport',                      # port strings / identifiers
        'Packet_num',                          # row index
        'Label',                               # keep 'Attack Category' only
    ],

    # Categorical features that need label/paper encoding
    'categorical_features': ['Flgs'],

    # Numeric features with heavy-tailed distribution -> log10(x + 1)
    'log_features': [
        'SrcBytes', 'DstBytes', 'TotBytes',
        'SrcLoad', 'DstLoad', 'Load', 'Rate',
        'SrcJitter', 'DstJitter',
        'SIntPkt', 'DIntPkt', 'SIntPktAct', 'DIntPktAct',
        'SrcGap', 'DstGap',
        'Dur', 'TotPkts', 'Loss',
        'sMaxPktSz', 'dMaxPktSz', 'sMinPktSz', 'dMinPktSz',
    ],

    # Features already in [0, 1] -> simple multiply by 255.
    # NOTE: in WUSTL-EHMS the pLoss / pSrcLoss / pDstLoss columns are percentages
    # up to ~33, not ratios in [0, 1], so they are intentionally left out here
    # and are handled by the generic MinMax normalizer instead.
    'rate_features': [],

    # Plain binary 0/1 features -> 0 stays 0, 1 -> 255
    'binary_features': [],

    # Discrete small-cardinality features (e.g. 0/1/2) -> 85/170/255
    'discrete_features': [],

    # Biometric / remaining numeric features (handled by the generic min-max normalizer)
    'biometric_features': ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA',
                           'Heart_rate', 'Resp_Rate', 'ST'],
}

# ---------------------------------------------------------------------------
# Legacy module-level aliases (kept so the existing pipeline functions below,
# which were written for NSL-KDD, keep working without further refactoring).
# ---------------------------------------------------------------------------
LOW_INFORMATION_FEATURES = CONFIG['drop_features']
LOG_SCALING_FEATURES = CONFIG['log_features']
BINARY_FEATURES = CONFIG['binary_features']
DISCRETE_FEATURES = CONFIG['discrete_features']
RATE_FEATURES = CONFIG['rate_features']

# Paper-style deterministic mappings for the WUSTL-EHMS categorical columns.
# Values are spread across [0, 255] so the encoded column is directly pixel-ready.
# Note: Flgs values in the raw CSV have padding whitespace (e.g. ' e        ');
# encoding strips whitespace first, so keys here are the trimmed forms.
PAPER_MAPPINGS = {
    'Flgs': {
        'e':   32,
        'M':   64,
        'eR':  96,
        'M *': 128,
        'e s': 160,
        'M d': 192,
        'MR':  224,
    },
}


def load_dataset(file_path):
    """
    Load the WUSTL-EHMS dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
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
    """
    Remove features with low information gain based on algorithm.

    Args:
        df (pd.DataFrame): Input dataframe
        features_to_remove (list): List of feature names to remove

    Returns:
        pd.DataFrame: Dataframe with features removed
    """
    print(f"Removing {len(features_to_remove)} low information gain features...")

    # Check which features actually exist in the dataframe
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


def apply_log_scaling(df, log_features):
    """
    Apply log10 scaling to specified numerical features.

    Args:
        df (pd.DataFrame): Input dataframe
        log_features (list): List of features to apply log scaling

    Returns:
        pd.DataFrame: Dataframe with log scaled features
    """
    print("Applying log scaling to numerical features...")

    df_scaled = df.copy()
    existing_log_features = [f for f in log_features if f in df.columns]
    missing_log_features = [f for f in log_features if f not in df.columns]

    if missing_log_features:
        print(f"  Warning: Log scaling features not found: {missing_log_features}")

    if existing_log_features:
        for feature in existing_log_features:
            # Apply log10(x + 1) to handle zero values
            original_min = df_scaled[feature].min()
            original_max = df_scaled[feature].max()
            df_scaled[feature] = np.log10(df_scaled[feature] + 1)
            new_min = df_scaled[feature].min()
            new_max = df_scaled[feature].max()
            print(f"    {feature}: [{original_min:.2f}, {original_max:.2f}] -> [{new_min:.4f}, {new_max:.4f}]")

        print(f"  ✓ Log scaling applied to {len(existing_log_features)} features")
    else:
        print("  No features found for log scaling")

    return df_scaled


def create_paper_compatible_encoder(feature_name, unique_values, paper_mapping):
    """
    Create a LabelEncoder that produces mappings compatible with the paper's specifications.

    Args:
        feature_name (str): Name of the feature being encoded
        unique_values (array): Unique values found in the dataset
        paper_mapping (dict): Paper-specific mapping dictionary

    Returns:
        LabelEncoder: Configured encoder
        dict: Actual mapping used
    """
    encoder = LabelEncoder()

    # Check if all unique values exist in paper mapping
    missing_values = set(unique_values) - set(paper_mapping.keys())
    if missing_values:
        print(f"    Warning: Values not in paper mapping for {feature_name}: {missing_values}")
        # Create mapping for missing values
        max_paper_value = max(paper_mapping.values()) if paper_mapping else 0
        for i, missing_val in enumerate(sorted(missing_values)):
            paper_mapping[missing_val] = max_paper_value + (i + 1) * 10

    # Create ordered list based on paper mapping values
    sorted_values = sorted(unique_values, key=lambda x: paper_mapping.get(x, float('inf')))

    # Fit encoder with sorted values to ensure consistent ordering
    encoder.fit(sorted_values)

    # Create the actual mapping that will be applied
    actual_mapping = {}
    for value in unique_values:
        if value in paper_mapping:
            actual_mapping[value] = paper_mapping[value]
        else:
            # Fallback for unmapped values
            encoded_label = encoder.transform([value])[0]
            actual_mapping[value] = encoded_label

    return encoder, actual_mapping


def apply_automated_categorical_encoding(df, categorical_features=None):
    """
    Apply automated label encoding to categorical features using scikit-learn's LabelEncoder
    while maintaining paper-specific mappings from PAPER_MAPPINGS.

    Args:
        df (pd.DataFrame): Input dataframe
        categorical_features (list): Columns to encode. Defaults to
            CONFIG['categorical_features'].

    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    print("Applying automated categorical encoding with paper-specific mappings...")

    if categorical_features is None:
        categorical_features = CONFIG.get('categorical_features', [])

    df_encoded = df.copy()
    encoding_summary = {}

    for feature in categorical_features:
        if feature not in df_encoded.columns:
            print(f"  Warning: Categorical feature '{feature}' not found, skipping")
            continue

        print(f"  Encoding {feature}...")

        # Strip whitespace for string-like columns (WUSTL-EHMS 'Flgs' contains
        # padding spaces). We check for "not numeric" to also cover the pandas
        # StringDtype, which is neither object nor a numpy str_ dtype.
        if not pd.api.types.is_numeric_dtype(df_encoded[feature]):
            df_encoded[feature] = df_encoded[feature].astype(str).str.strip()

        unique_values = df_encoded[feature].unique()
        print(f"    Unique values ({len(unique_values)}): {sorted(unique_values)[:20]}"
              + (" ..." if len(unique_values) > 20 else ""))

        paper_mapping = dict(PAPER_MAPPINGS.get(feature, {}))
        encoder, mapping = create_paper_compatible_encoder(
            feature, unique_values, paper_mapping
        )

        df_encoded[feature] = df_encoded[feature].map(mapping)
        encoding_summary[feature] = mapping

        unmapped = df_encoded[feature].isna().sum()
        if unmapped > 0:
            print(f"    Warning: {unmapped} unmapped {feature} values, filling with 0")
            df_encoded[feature].fillna(0, inplace=True)

        if len(mapping) <= 20:
            print(f"    ✓ {feature} mapping applied: {mapping}")
        else:
            print(f"    ✓ {feature} mapping applied ({len(mapping)} entries)")
        print(f"    Value distribution: {df_encoded[feature].value_counts().to_dict()}")

    print("  ✓ Automated categorical encoding completed")

    save_encoding_summary(encoding_summary)

    return df_encoded


def save_encoding_summary(encoding_summary):
    """
    Save the encoding summary to a file for reference.

    Args:
        encoding_summary (dict): Dictionary containing encoding mappings
    """
    output_file = os.path.join(CONFIG['output_dir'], 'encoding_summary.txt')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Categorical Feature Encoding Summary\n")
        f.write("=" * 50 + "\n\n")

        for feature_name, mapping in encoding_summary.items():
            f.write(f"{feature_name.upper()}:\n")
            f.write("-" * 20 + "\n")
            for original, encoded in sorted(mapping.items(), key=lambda x: x[1]):
                f.write(f"  {original} -> {encoded}\n")
            f.write("\n")

    print(f"  ✓ Encoding summary saved: {output_file}")


def apply_binary_mapping(df, binary_features):
    """
    Map binary features: 0 -> 0, 1 -> 255.

    Args:
        df (pd.DataFrame): Input dataframe
        binary_features (list): List of binary features

    Returns:
        pd.DataFrame: Dataframe with binary mapping applied
    """
    print("Applying binary mapping (0->0, 1->255)...")

    df_mapped = df.copy()
    existing_binary = [f for f in binary_features if f in df.columns]
    missing_binary = [f for f in binary_features if f not in df.columns]

    if missing_binary:
        print(f"  Warning: Binary features not found: {missing_binary}")

    if existing_binary:
        for feature in existing_binary:
            original_dist = df_mapped[feature].value_counts().to_dict()
            df_mapped.loc[df_mapped[feature] == 1, feature] = 255
            # 0 values remain 0
            new_dist = df_mapped[feature].value_counts().to_dict()

            print(f"    {feature}: {original_dist} -> {new_dist}")

        print(f"  ✓ Binary mapping applied to {len(existing_binary)} features")
    else:
        print("  No binary features found for mapping")

    return df_mapped


def apply_discrete_mapping(df, discrete_features):
    """
    Map discrete features: 0->85, 1->170, 2->255.

    Args:
        df (pd.DataFrame): Input dataframe
        discrete_features (list): List of discrete features

    Returns:
        pd.DataFrame: Dataframe with discrete mapping applied
    """
    print("Applying discrete mapping (0->85, 1->170, 2->255)...")

    df_mapped = df.copy()
    existing_discrete = [f for f in discrete_features if f in df.columns]
    missing_discrete = [f for f in discrete_features if f not in df.columns]

    if missing_discrete:
        print(f"  Warning: Discrete features not found: {missing_discrete}")

    if existing_discrete:
        for feature in existing_discrete:
            original_dist = df_mapped[feature].value_counts().to_dict()

            df_mapped.loc[df_mapped[feature] == 0, feature] = 85
            df_mapped.loc[df_mapped[feature] == 1, feature] = 170
            df_mapped.loc[df_mapped[feature] == 2, feature] = 255

            new_dist = df_mapped[feature].value_counts().to_dict()
            print(f"    {feature}: {original_dist} -> {new_dist}")

        print(f"  ✓ Discrete mapping applied to {len(existing_discrete)} features")
    else:
        print("  No discrete features found for mapping")

    return df_mapped


def apply_rate_scaling(df, rate_features):
    """
    Scale rate features by multiplying with 255 (normalize to [0, 255] range).

    Args:
        df (pd.DataFrame): Input dataframe
        rate_features (list): List of rate features

    Returns:
        pd.DataFrame: Dataframe with rate scaling applied
    """
    print("Applying rate scaling (multiply by 255)...")

    df_scaled = df.copy()
    existing_rate = [f for f in rate_features if f in df.columns]
    missing_rate = [f for f in rate_features if f not in df.columns]

    if missing_rate:
        print(f"  Warning: Rate features not found: {missing_rate}")

    if existing_rate:
        for feature in existing_rate:
            original_range = (df_scaled[feature].min(), df_scaled[feature].max())
            df_scaled[feature] = df_scaled[feature] * 255
            new_range = (df_scaled[feature].min(), df_scaled[feature].max())
            print(
                f"    {feature}: [{original_range[0]:.4f}, {original_range[1]:.4f}] -> [{new_range[0]:.2f}, {new_range[1]:.2f}]")

        print(f"  ✓ Rate scaling applied to {len(existing_rate)} features")
    else:
        print("  No rate features found for scaling")

    return df_scaled


def normalize_remaining_features(df, exclude_features=None):
    """
    Normalize remaining numerical features to [0, 255] range using log scaling + min-max scaling.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude_features (list): Features to exclude from normalization

    Returns:
        pd.DataFrame: Dataframe with normalized features
    """
    if exclude_features is None:
        exclude_features = []

    print("Normalizing remaining numerical features to [0, 255] range...")

    df_normalized = df.copy()
    target_column = CONFIG.get('target_column')

    processed_features = (LOG_SCALING_FEATURES + BINARY_FEATURES +
                          DISCRETE_FEATURES + RATE_FEATURES +
                          list(CONFIG.get('categorical_features', [])) +
                          ([target_column] if target_column else []) +
                          exclude_features)

    remaining_features = []
    for col in df_normalized.columns:
        if (col not in processed_features and
                df_normalized[col].dtype in ['int64', 'float64'] and
                col != target_column):
            remaining_features.append(col)

    if remaining_features:
        print(f"  Features to normalize: {remaining_features}")

        for feature in remaining_features:
            original_range = (df_normalized[feature].min(), df_normalized[feature].max())

            # Apply log scaling first if the range is large
            if original_range[1] > 1000:
                df_normalized[feature] = np.log10(df_normalized[feature] + 1)

            # Apply min-max scaling to [0, 255] range
            scaler = MinMaxScaler(feature_range=(0, 255))
            df_normalized[feature] = scaler.fit_transform(df_normalized[[feature]]).flatten()

            new_range = (df_normalized[feature].min(), df_normalized[feature].max())
            print(
                f"    {feature}: [{original_range[0]:.2f}, {original_range[1]:.2f}] -> [{new_range[0]:.2f}, {new_range[1]:.2f}]")

        print(f"  ✓ Normalization applied to {len(remaining_features)} features")
    else:
        print("  No remaining features found for normalization")

    return df_normalized


def convert_to_binary_classification(df, target_column=None, attack_classes=None,
                                     normal_class=None):
    """
    Convert multi-class classification to binary (normal vs attack).

    All labels in `attack_classes` are collapsed to 'attack'; everything else is
    left untouched (typically only the 'normal' class remains).

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column. Defaults to
            CONFIG['target_column'].
        attack_classes (list): Labels to be relabelled as 'attack'. Defaults to
            CONFIG['attack_classes'].
        normal_class (str): Name of the benign class. Defaults to
            CONFIG['normal_class'].

    Returns:
        pd.DataFrame: Dataframe with binary classification
    """
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

    # Normalise the benign label to lowercase 'normal' for downstream consistency.
    if normal_class != 'normal':
        df_binary.loc[df_binary[target_column] == normal_class, target_column] = 'normal'

    print(f"  ✓ Binary classification applied")
    print(f"  New class distribution:")
    for class_name, count in df_binary[target_column].value_counts().items():
        print(f"    {class_name}: {count}")

    return df_binary


def convert_data_types(df):
    """
    Convert float64 columns to int64 for consistency.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with converted data types
    """
    print("Converting data types...")

    df_converted = df.copy()
    target_column = CONFIG.get('target_column')
    float_columns = []

    for column in df_converted.columns:
        if (df_converted[column].dtype == 'float64' and
                column != target_column):
            # Check if all values are close to integers
            if df_converted[column].notna().all():
                # Round to nearest integer and convert
                df_converted[column] = np.round(df_converted[column]).astype('int64')
                float_columns.append(column)

    if float_columns:
        print(f"  ✓ Converted {len(float_columns)} columns from float64 to int64")
    else:
        print("  No columns needed conversion")

    return df_converted


def save_preprocessed_data(df, output_dir, filename_prefix="automated_label_preprocessed",
                           dataset_tag="wustlehms"):
    """
    Save preprocessed data to CSV files.

    Args:
        df (pd.DataFrame): Preprocessed dataframe
        output_dir (str): Output directory
        filename_prefix (str): Prefix for output filenames
        dataset_tag (str): Short dataset identifier appended to filenames
    """
    print("Saving preprocessed data...")

    os.makedirs(output_dir, exist_ok=True)
    target_column = CONFIG.get('target_column')

    complete_file = os.path.join(output_dir, f"{filename_prefix}_{dataset_tag}.csv")
    df.to_csv(complete_file, index=False)
    print(f"  ✓ Complete dataset saved: {complete_file}")

    if target_column and target_column in df.columns:
        normal_df = df[df[target_column] == 'normal']
        attack_df = df[df[target_column] == 'attack']

        normal_file = os.path.join(output_dir, f"{filename_prefix}_{dataset_tag}_normal.csv")
        attack_file = os.path.join(output_dir, f"{filename_prefix}_{dataset_tag}_attack.csv")

        normal_df.to_csv(normal_file, index=False)
        attack_df.to_csv(attack_file, index=False)

        print(f"  ✓ Normal data saved: {normal_file} ({len(normal_df)} samples)")
        print(f"  ✓ Attack data saved: {attack_file} ({len(attack_df)} samples)")

    feature_file = os.path.join(output_dir, f"{filename_prefix}_{dataset_tag}_feature_list.txt")
    feature_columns = [col for col in df.columns if col != target_column]
    with open(feature_file, 'w') as f:
        f.write("Feature List (excluding target column):\n")
        f.write("=" * 50 + "\n")
        for i, feature in enumerate(feature_columns, 1):
            f.write(f"{i:2d}. {feature}\n")

    print(f"  ✓ Feature list saved: {feature_file}")


def display_preprocessing_summary(df_original, df_final):
    """
    Display a summary of the preprocessing transformations.

    Args:
        df_original (pd.DataFrame): Original dataframe
        df_final (pd.DataFrame): Final preprocessed dataframe
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Original dataset shape: {df_original.shape}")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Features removed: {df_original.shape[1] - df_final.shape[1]}")

    print(f"\nData type distribution:")
    for dtype in df_final.dtypes.value_counts().index:
        count = df_final.dtypes.value_counts()[dtype]
        print(f"  {dtype}: {count} columns")

    target_column = CONFIG.get('target_column')
    print(f"\nValue range summary (excluding target column '{target_column}'):")
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]

    if len(numeric_cols) > 0:
        print(f"  Minimum value: {df_final[numeric_cols].min().min():.2f}")
        print(f"  Maximum value: {df_final[numeric_cols].max().max():.2f}")
        print(
            f"  Features in [0, 255] range: {len([col for col in numeric_cols if df_final[col].min() >= 0 and df_final[col].max() <= 255])}/{len(numeric_cols)}")


def main():
    """
    Main function to run the complete automated label encoding preprocessing pipeline.
    """
    print("=" * 60)
    print("WUSTL-EHMS Dataset Preprocessing with Automated Label Encoding")
    print("=" * 60)

    try:
        # Step 1: Load dataset
        df_original = load_dataset(CONFIG['input_file'])
        df = df_original.copy()

        # Step 2: Remove low information gain features
        df = remove_low_information_features(df, LOW_INFORMATION_FEATURES)

        # Step 3: Apply log scaling to numerical features
        df = apply_log_scaling(df, LOG_SCALING_FEATURES)

        # Step 4: Apply automated categorical encoding (replaces manual encoding)
        df = apply_automated_categorical_encoding(df)

        # Step 5: Apply binary mapping
        df = apply_binary_mapping(df, BINARY_FEATURES)

        # Step 6: Apply discrete mapping
        df = apply_discrete_mapping(df, DISCRETE_FEATURES)

        # Step 7: Apply rate scaling
        df = apply_rate_scaling(df, RATE_FEATURES)

        # Step 8: Normalize remaining features
        df = normalize_remaining_features(df)

        # Step 9: Convert to binary classification
        df = convert_to_binary_classification(df)

        # Step 10: Convert data types
        df = convert_data_types(df)

        # Step 11: Save preprocessed data
        save_preprocessed_data(df, CONFIG['output_dir'], "automated_label_preprocessed")

        # Step 12: Display summary
        display_preprocessing_summary(df_original, df)

        print("\n" + "=" * 60)
        print("✓ Automated label encoding preprocessing completed successfully!")
        print(f"✓ Final dataset shape: {df.shape}")
        print(f"✓ Output saved to: {CONFIG['output_dir']}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()