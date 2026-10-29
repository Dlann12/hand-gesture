"""
Hand landmark extraction module using MediaPipe Hands solution.
Extracts 21 landmarks (x, y, z coordinates) from camera or image frames.
"""

from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

LANDMARK_FEATURE_NAMES = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


class HandLandmarkExtractor:
    """
    Extracts 21 3D hand landmarks using MediaPipe Hands.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(
        self,
        image_bgr: np.ndarray,
        draw: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[any]]:
        """
        Extract landmark coordinates from a BGR image (OpenCV format).

        Args:
            image_bgr: Input image in BGR format.
            draw: Whether to render skeleton landmarks on the output annotated image.

        Returns:
            Tuple of:
            - landmarks: np.ndarray of shape (63,) with [x0, y0, z0, ..., x20, y20, z20] or None if not found.
            - annotated_image: Image with drawn landmarks (BGR format).
            - raw_hand_landmarks: MediaPipe LandmarkList object or None.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None, image_bgr, None

        annotated_image = image_bgr.copy()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None, annotated_image, None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract 63 raw coordinate values
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        landmarks_array = np.array(coords, dtype=np.float32)

        if draw:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        return landmarks_array, annotated_image, hand_landmarks

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
