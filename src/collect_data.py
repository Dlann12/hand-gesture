"""
Interactive webcam data collector for real hand gesture landmarks.
Extracts 21 3D landmarks in real time and appends to CSV files.

Usage:
    python src/collect_data.py --label 1 --samples 300
    
Key controls:
    [SPACE]   Capture 1 sample
    [A]       Toggle automatic capture mode (captures periodically)
    [Q / ESC] Quit collector
"""

import argparse
import csv
from pathlib import Path
import time
import cv2
import mediapipe as mp

from src.extract_landmarks import LANDMARK_FEATURE_NAMES

COLUMNS = ["label", "gesture", *LANDMARK_FEATURE_NAMES, "sample_id"]


def main():
    parser = argparse.ArgumentParser(description="Collect real hand gesture landmarks from webcam")
    parser.add_argument("--label", type=int, required=True, choices=range(6), help="Gesture label (0 to 5)")
    parser.add_argument("--samples", type=int, default=300, help="Target number of samples to collect")
    parser.add_argument("--out", default="data/raw", help="Output directory for CSV files")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"gestures_{args.label}.csv"

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(
            f"Tidak dapat mengakses kamera index {args.cam}. "
            "Pastikan webcam terhubung dan tidak sedang digunakan aplikasi lain."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    captured = 0
    auto_mode = False
    last_auto_capture = 0.0

    print("=" * 60)
    print(f"Mengumpulkan Landmark Gestur Angka: {args.label}")
    print(f"Target: {args.samples} sampel -> {out_file}")
    print("Kontrol:")
    print("  [SPACE] : Ambil 1 sampel")
    print("  [A]     : Aktifkan / Matikan mode Auto-capture")
    print("  [Q/ESC] : Berhenti dan keluar")
    print("=" * 60)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    ) as hands:
        while captured < args.samples:
            ok, frame = cap.read()
            if not ok:
                print("Peringatan: Gagal membaca frame dari webcam.")
                break

            frame = cv2.flip(frame, 1)  # Mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            display = frame.copy()

            hand_found = bool(result.multi_hand_landmarks)
            trigger_capture = False

            if hand_found:
                hand = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)

                # Status bar
                mode_str = "AUTO" if auto_mode else "MANUAL [SPACE]"
                cv2.putText(
                    display,
                    f"Label: {args.label} | {captured}/{args.samples} | Mode: {mode_str}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                now = time.time()
                if auto_mode and (now - last_auto_capture >= 0.15):
                    trigger_capture = True
                    last_auto_capture = now
            else:
                cv2.putText(
                    display,
                    "Tangan Tidak Terdeteksi",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Hand Gesture Data Collector", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                print("Perekaman dihentikan oleh pengguna.")
                break
            elif key == ord("a") or key == ord("A"):
                auto_mode = not auto_mode
                print(f"Mode Auto-capture: {'AKTIF' if auto_mode else 'NONAKTIF'}")
            elif key == 32:  # SPACE
                if hand_found:
                    trigger_capture = True
                else:
                    print("Tangan tidak terdeteksi, tidak dapat menyimpan sampel.")

            if trigger_capture and hand_found:
                row = {
                    "label": args.label,
                    "gesture": f"number_{args.label}",
                    "sample_id": f"real_{args.label}_{int(time.time() * 1000)}_{captured}"
                }
                for i, p in enumerate(hand.landmark):
                    row[f"x{i}"] = p.x
                    row[f"y{i}"] = p.y
                    row[f"z{i}"] = p.z

                write_header = not out_file.exists() or out_file.stat().st_size == 0
                with out_file.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                captured += 1
                if captured % 25 == 0 or captured == args.samples:
                    print(f"Tersimpan: {captured}/{args.samples} sampel")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelesai! Berhasil mengumpulkan {captured} sampel ke: {out_file}")


if __name__ == "__main__":
    main()
