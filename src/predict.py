"""
Inference and prediction module for Hand Gesture Recognition.
Loads trained classifier, normalizes input landmarks, applies confidence thresholds,
and returns rich prediction results with confidence scores.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import joblib
import numpy as np

from src.extract_landmarks import HandLandmarkExtractor
from src.preprocess import normalize_landmarks

DEFAULT_LABELS = {
    0: "Tangan Mengepal (0)",
    1: "Satu Jari Terangkat (1)",
    2: "Dua Jari Terangkat (2)",
    3: "Tiga Jari Terangkat (3)",
    4: "Empat Jari Terangkat (4)",
    5: "Lima Jari Terbuka (5)"
}

DEFAULT_GESTURE_NAMES = {
    0: "number_0",
    1: "number_1",
    2: "number_2",
    3: "number_3",
    4: "number_4",
    5: "number_5"
}


class GesturePredictor:
    """
    Predicts hand gesture (number 0-5) from raw 21 3D landmarks or camera frames.
    """

    def __init__(
        self,
        model_path: str = "models/gesture_classifier.joblib",
        metadata_path: Optional[str] = "models/metadata.json",
        confidence_threshold: float = 0.60
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.confidence_threshold = float(confidence_threshold)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Please run training first: 'python src/train.py'"
            )

        self.model = joblib.load(self.model_path)
        self.metadata = {}
        if self.metadata_path and self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Load class labels from metadata if available
        labels_meta = self.metadata.get("class_labels", {})
        if labels_meta:
            self.labels = {int(k): v for k, v in labels_meta.items()}
        else:
            self.labels = DEFAULT_LABELS

        gestures_meta = self.metadata.get("class_gestures", {})
        if gestures_meta:
            self.gestures = {int(k): v for k, v in gestures_meta.items()}
        else:
            self.gestures = DEFAULT_GESTURE_NAMES

        # Lazy initialized extractor
        self._extractor: Optional[HandLandmarkExtractor] = None

    @property
    def extractor(self) -> HandLandmarkExtractor:
        if self._extractor is None:
            self._extractor = HandLandmarkExtractor(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65
            )
        return self._extractor

    def predict_landmarks(
        self,
        raw_landmarks: Union[np.ndarray, List[float]]
    ) -> Dict[str, Any]:
        """
        Predict hand gesture number from 63 raw landmarks.

        Args:
            raw_landmarks: 63 floats (21 landmarks x 3 coordinates).

        Returns:
            Dictionary containing prediction, probabilities, confidence, and status.
        """
        # Step 1: Apply translation and scale normalization
        normalized = normalize_landmarks(raw_landmarks)

        # Step 2: Model prediction
        features_2d = normalized.reshape(1, -1)
        probabilities = self.model.predict_proba(features_2d)[0]
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        # Step 3: Confidence threshold check (Unknown rejection)
        is_unknown = confidence < self.confidence_threshold
        predicted_label = -1 if is_unknown else predicted_idx
        display_name = "Unknown" if is_unknown else self.labels.get(predicted_idx, f"Angka {predicted_idx}")
        gesture_code = "unknown" if is_unknown else self.gestures.get(predicted_idx, f"number_{predicted_idx}")

        prob_dict = {
            int(i): float(probabilities[i]) if i < len(probabilities) else 0.0
            for i in range(len(self.labels))
        }

        return {
            "predicted_label": predicted_label,
            "display_name": display_name,
            "gesture_code": gesture_code,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "is_unknown": is_unknown,
            "class_probabilities": prob_dict,
            "normalized_features": normalized.tolist()
        }

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        draw_overlay: bool = True
    ) -> Dict[str, Any]:
        """
        Extract hand landmarks from image frame and predict gesture.

        Args:
            image_bgr: OpenCV BGR image frame.
            draw_overlay: Whether to draw landmarks and prediction label overlay on frame.

        Returns:
            Prediction result dictionary including annotated frame.
        """
        landmarks_raw, annotated_image, hand_landmarks = self.extractor.extract(
            image_bgr, draw=draw_overlay
        )

        if landmarks_raw is None:
            return {
                "hand_detected": False,
                "predicted_label": None,
                "display_name": "Tangan Tidak Terdeteksi",
                "gesture_code": "no_hand",
                "confidence": 0.0,
                "confidence_percentage": 0.0,
                "is_unknown": True,
                "class_probabilities": {i: 0.0 for i in range(len(self.labels))},
                "annotated_image": annotated_image,
                "raw_landmarks": None
            }

        pred = self.predict_landmarks(landmarks_raw)
        pred["hand_detected"] = True
        pred["raw_landmarks"] = landmarks_raw.tolist()

        if draw_overlay:
            h, w = annotated_image.shape[:2]

            if pred["is_unknown"]:
                badge_title = "ANGKA: ?"
                badge_color = (0, 165, 255)  # Orange
                status_text = f"Keyakinan: {pred['confidence_percentage']}% (Ragu-ragu)"
            else:
                num = pred["predicted_label"]
                badge_title = f"ANGKA: {num}"
                badge_color = (0, 255, 128)  # Bright neon green
                status_text = f"Keyakinan: {pred['confidence_percentage']}%"

            # Top-left HUD panel
            cv2.rectangle(annotated_image, (15, 15), (320, 85), (20, 24, 32), -1)
            cv2.rectangle(annotated_image, (15, 15), (320, 85), badge_color, 2)
            cv2.putText(
                annotated_image,
                badge_title,
                (28, 52),
                cv2.FONT_HERSHEY_DUPLEX,
                1.15,
                badge_color,
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                annotated_image,
                status_text,
                (28, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 225, 235),
                1,
                cv2.LINE_AA
            )

            # Draw floating number near palm if landmark available
            if hand_landmarks and len(hand_landmarks.landmark) > 9:
                palm_x = int(hand_landmarks.landmark[9].x * w)
                palm_y = int(hand_landmarks.landmark[9].y * h) - 25
                if 20 <= palm_x < w - 20 and 30 <= palm_y < h:
                    display_char = "?" if pred["is_unknown"] else str(pred["predicted_label"])
                    # Floating circular badge on hand
                    cv2.circle(annotated_image, (palm_x, palm_y), 22, (20, 24, 32), -1)
                    cv2.circle(annotated_image, (palm_x, palm_y), 22, badge_color, 2)
                    cv2.putText(
                        annotated_image,
                        display_char,
                        (palm_x - 9, palm_y + 8),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.85,
                        badge_color,
                        2,
                        cv2.LINE_AA
                    )

        pred["annotated_image"] = annotated_image
        return pred

    def close(self):
        """Release extractor resources."""
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None
