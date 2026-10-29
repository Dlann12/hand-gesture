"""
Standalone Native Desktop Live Webcam Hand Gesture Number Recognition (0–5).
Runs directly in an OpenCV window with zero web overhead at max FPS.

Usage:
    python live_cam.py
    python live_cam.py --cam 0 --threshold 0.60

Controls:
    [Q / ESC] : Keluar dari program
"""

import argparse
import time
import cv2

from src.predict import GesturePredictor


def main():
    parser = argparse.ArgumentParser(description="Live Hand Gesture Number Recognition (0-5)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold (default: 0.60)")
    parser.add_argument("--model", default="models/gesture_classifier.joblib", help="Model path")
    parser.add_argument("--metadata", default="models/metadata.json", help="Metadata path")
    args = parser.parse_args()

    print("=" * 60)
    print("🖐️ LIVE HAND GESTURE NUMBER RECOGNITION (0-5)")
    print(f"Kamera: {args.cam} | Ambang Confidence: {int(args.threshold * 100)}%")
    print("Tekan [Q] atau [ESC] pada jendela kamera untuk keluar.")
    print("=" * 60)

    predictor = GesturePredictor(
        model_path=args.model,
        metadata_path=args.metadata,
        confidence_threshold=args.threshold
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka webcam pada index {args.cam}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca frame.")
            break

        frame = cv2.flip(frame, 1)  # Cermin
        t0 = time.time()
        result = predictor.predict_frame(frame, draw_overlay=True)
        latency_ms = (time.time() - t0) * 1000
        fps = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()

        display = result["annotated_image"]

        # Status footer bar
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h - 35), (w, h), (15, 18, 24), -1)

        if not result["hand_detected"]:
            status_str = "Tangan: Tidak Terdeteksi | Tunjukkan jari 0-5"
            status_col = (130, 145, 160)
        elif result["is_unknown"]:
            status_str = f"Status: Unknown (Ragu-ragu) | Keyakinan: {result['confidence_percentage']}%"
            status_col = (0, 165, 255)
        else:
            status_str = f"ANGKA: {result['predicted_label']} | Keyakinan: {result['confidence_percentage']}%"
            status_col = (0, 255, 128)

        cv2.putText(
            display,
            f"{status_str} | {fps:.0f} FPS ({latency_ms:.1f}ms)",
            (15, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_col,
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Hand Gesture Recognition - Live 0 to 5", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    predictor.close()
    print("Program selesai.")


if __name__ == "__main__":
    main()
