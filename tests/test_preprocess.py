"""
Unit tests for preprocessing and landmark normalization.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    LANDMARK_FEATURE_NAMES,
    normalize_landmarks,
    split_dataset
)


def create_dummy_landmarks(scale: float = 1.0, offset: np.ndarray = None) -> np.ndarray:
    """Generate dummy 21 3D landmarks."""
    coords = np.arange(63, dtype=np.float32).reshape(21, 3) * scale
    if offset is not None:
        coords += offset
    return coords.flatten()


def test_normalize_landmarks_shape_1d():
    raw = create_dummy_landmarks()
    norm = normalize_landmarks(raw)
    assert norm.shape == (63,)
    # Wrist (index 0) must be centered at origin (0, 0, 0)
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[1], 0.0)
    assert np.isclose(norm[2], 0.0)


def test_normalize_landmarks_translation_invariance():
    """Shifting hand position by offset (dx, dy, dz) should produce identical normalized features."""
    raw1 = create_dummy_landmarks(offset=np.array([0.0, 0.0, 0.0]))
    raw2 = create_dummy_landmarks(offset=np.array([150.0, -80.0, 42.0]))

    norm1 = normalize_landmarks(raw1)
    norm2 = normalize_landmarks(raw2)

    np.testing.assert_allclose(norm1, norm2, atol=1e-5)


def test_normalize_landmarks_scale_invariance():
    """Scaling hand size by factor s should produce identical normalized features."""
    raw1 = create_dummy_landmarks(scale=1.0)
    raw2 = create_dummy_landmarks(scale=4.5)

    norm1 = normalize_landmarks(raw1)
    norm2 = normalize_landmarks(raw2)

    np.testing.assert_allclose(norm1, norm2, atol=1e-5)


def test_normalize_landmarks_batch_2d():
    batch = np.array([
        create_dummy_landmarks(scale=1.0, offset=np.array([10.0, 20.0, 30.0])),
        create_dummy_landmarks(scale=2.0, offset=np.array([-5.0, 15.0, 0.0]))
    ])
    norm_batch = normalize_landmarks(batch)
    assert norm_batch.shape == (2, 63)
    # Both wrist landmarks should be (0, 0, 0)
    np.testing.assert_allclose(norm_batch[:, :3], 0.0, atol=1e-6)


def test_invalid_shape_raises_error():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros(50))


def test_split_dataset_stratification():
    # Create balanced dataset of 600 samples (100 per class 0..5)
    data = []
    for label in range(6):
        for i in range(100):
            row = {col: np.random.rand() for col in LANDMARK_FEATURE_NAMES}
            row["label"] = label
            row["gesture"] = f"number_{label}"
            data.append(row)
    df = pd.DataFrame(data)

    train_df, val_df, test_df = split_dataset(
        df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )

    assert len(train_df) == 420
    assert len(val_df) == 90
    assert len(test_df) == 90

    # Ensure all classes are present in exact stratified ratios
    for label in range(6):
        assert (train_df["label"] == label).sum() == 70
        assert (val_df["label"] == label).sum() == 15
        assert (test_df["label"] == label).sum() == 15
