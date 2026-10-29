"""
Data preprocessing and landmark normalization module.
Ensures scale and translation invariance for hand landmarks.
"""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LANDMARK_FEATURE_NAMES = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def normalize_landmarks(
    landmarks: Union[np.ndarray, List[float]],
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Normalize 21 hand landmarks (63 values) to be translation and scale invariant.
    
    1. Translation invariance: Wrist landmark (index 0) is set as origin (0, 0, 0).
    2. Scale invariance: All coordinates are divided by the maximum distance from wrist.

    Args:
        landmarks: Array-like of shape (63,), (21, 3), (N, 63), or (N, 21, 3).
        epsilon: Small constant to prevent division by zero.

    Returns:
        Normalized landmarks array matching input batch dimension:
        - If input was 1D (63,) -> returns (63,)
        - If input was (N, 63) -> returns (N, 63)
    """
    arr = np.asarray(landmarks, dtype=np.float32)
    original_shape = arr.shape

    # Handle single sample 1D
    is_single_sample = False
    if arr.ndim == 1:
        if arr.shape[0] != 63:
            raise ValueError(f"Expected 63 landmark coordinates, got {arr.shape[0]}")
        arr = arr.reshape(1, 21, 3)
        is_single_sample = True
    elif arr.ndim == 2:
        if arr.shape == (21, 3):
            arr = arr.reshape(1, 21, 3)
            is_single_sample = True
        elif arr.shape[1] == 63:
            arr = arr.reshape(-1, 21, 3)
        else:
            raise ValueError(f"Invalid 2D shape {arr.shape}. Expected (N, 63) or (21, 3)")
    elif arr.ndim == 3:
        if arr.shape[1:] != (21, 3):
            raise ValueError(f"Invalid 3D shape {arr.shape}. Expected (N, 21, 3)")
    else:
        raise ValueError(f"Unsupported array dimensions: {arr.ndim}")

    # Step 1: Center around wrist (landmark 0)
    wrist = arr[:, 0:1, :]  # Shape: (N, 1, 3)
    centered = arr - wrist   # Shape: (N, 21, 3)

    # Step 2: Compute euclidean distance of each landmark from wrist
    distances = np.linalg.norm(centered, axis=2)  # Shape: (N, 21)
    max_distances = np.max(distances, axis=1, keepdims=True)  # Shape: (N, 1)

    # Prevent division by zero
    max_distances = np.maximum(max_distances, epsilon)

    # Step 3: Scale normalization
    normalized = centered / max_distances[:, :, np.newaxis]

    # Flatten back to 63 features per sample
    flattened = normalized.reshape(arr.shape[0], 63)

    if is_single_sample and len(original_shape) == 1:
        return flattened[0]
    return flattened


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load landmark dataset from CSV file and validate schema.
    """
    df = pd.read_csv(csv_path)

    # Verify label exists
    if "label" not in df.columns:
        raise ValueError("Dataset missing required 'label' column")

    # Verify all 63 feature columns exist
    missing_cols = [col for col in LANDMARK_FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required landmark columns: {missing_cols[:5]}...")

    # Check for nulls
    if df[LANDMARK_FEATURE_NAMES].isnull().any().any():
        print("Warning: NaN values found in features. Dropping invalid rows...")
        df = df.dropna(subset=LANDMARK_FEATURE_NAMES + ["label"]).copy()

    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into stratified train, validation, and test subsets.
    Default: 70% train, 15% validation, 15% test.
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0 (got {total})")

    # Step 1: Separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    # Step 2: Separate train and validation sets from remainder
    val_fraction_of_remainder = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_remainder,
        stratify=train_val_df["label"],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def prepare_features_and_labels(
    df: pd.DataFrame,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from DataFrame.
    Applies landmark normalization if normalize=True.
    """
    raw_x = df[LANDMARK_FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(int)

    if normalize:
        x_norm = normalize_landmarks(raw_x)
        return x_norm, y
    return raw_x, y
