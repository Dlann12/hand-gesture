"""
Unit tests for gesture prediction module.
"""

from pathlib import Path
import numpy as np
import pytest

from src.predict import GesturePredictor
from src.preprocess import LANDMARK_FEATURE_NAMES, load_dataset


@pytest.fixture
def predictor():
    return GesturePredictor(
        model_path="models/gesture_classifier.joblib",
        metadata_path="models/metadata.json",
        confidence_threshold=0.60
    )


def test_predictor_initialization(predictor):
    assert predictor.model is not None
    assert len(predictor.labels) == 6
    assert 0 in predictor.labels
    assert 5 in predictor.labels


def test_predict_landmarks_valid_sample(predictor):
    df = load_dataset("data/processed/synthetic_landmarks_demo.csv")
    sample_0 = df[df["label"] == 0].iloc[0][LANDMARK_FEATURE_NAMES].values.astype(np.float32)

    res = predictor.predict_landmarks(sample_0)

    assert "predicted_label" in res
    assert "confidence" in res
    assert "display_name" in res
    assert "class_probabilities" in res
    assert res["predicted_label"] == 0
    assert res["confidence"] >= 0.60
    assert not res["is_unknown"]


def test_predict_landmarks_unknown_rejection():
    """Predictor should reject prediction and set is_unknown=True when confidence < threshold."""
    # Force high threshold to test rejection
    strict_predictor = GesturePredictor(
        model_path="models/gesture_classifier.joblib",
        metadata_path="models/metadata.json",
        confidence_threshold=0.9999
    )
    df = load_dataset("data/processed/synthetic_landmarks_demo.csv")
    sample = df.iloc[0][LANDMARK_FEATURE_NAMES].values.astype(np.float32)

    res = strict_predictor.predict_landmarks(sample)
    # Even if label was class X, strict threshold should label it unknown
    if res["confidence"] < 0.9999:
        assert res["is_unknown"] is True
        assert res["predicted_label"] == -1
        assert res["display_name"] == "Unknown"


def test_predict_frame_no_hand(predictor):
    # Black image frame without hand
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = predictor.predict_frame(blank_frame, draw_overlay=False)

    assert res["hand_detected"] is False
    assert res["predicted_label"] is None
    assert res["confidence"] == 0.0
