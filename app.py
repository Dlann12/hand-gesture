"""
Hand Gesture Recognition System - Web Application
Live Real-Time Number Detection (0–5)
"""

import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.extract_landmarks import HandLandmarkExtractor, LANDMARK_FEATURE_NAMES
from src.predict import GesturePredictor

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Angka Gestur Tangan (0–5) Live",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS focused on prominent number display
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&family=JetBrains+Mono:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Giant glowing hero number */
    .hero-badge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle at 50% 15%, rgba(14, 165, 233, 0.2) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid rgba(56, 189, 248, 0.4);
        border-radius: 24px;
        padding: 35px 20px;
        text-align: center;
        box-shadow: 0 20px 40px -10px rgba(14, 165, 233, 0.3);
        margin-bottom: 16px;
    }
    .hero-number {
        font-size: 7.5rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 900;
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 12px 0;
        filter: drop-shadow(0 0 35px rgba(56, 189, 248, 0.6));
    }
    .hero-number-unknown {
        font-size: 5.5rem;
        font-weight: 900;
        color: #f59e0b;
        margin: 12px 0;
        filter: drop-shadow(0 0 25px rgba(245, 158, 11, 0.5));
    }
    .hero-number-empty {
        font-size: 4.5rem;
        font-weight: 800;
        color: #64748b;
        margin: 12px 0;
    }
    .hero-label {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .hero-status-pill {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }
    .pill-orange {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.5);
    }
    .pill-gray {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Metric cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Quick guide pills */
    .gesture-guide-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching Resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_predictor(confidence_threshold: float = 0.60):
    return GesturePredictor(
        model_path="models/gesture_classifier.joblib",
        metadata_path="models/metadata.json",
        confidence_threshold=confidence_threshold
    )


# Initialize Session State
if "camera_running" not in st.session_state:
    st.session_state.camera_running = True


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🖐️ Kontrol Live Deteksi")
    st.caption("Deteksi gestur angka 0–5 secara live real-time.")
    st.markdown("---")

    cam_index = st.selectbox("Pilih Kamera Webcam:", [0, 1, 2], index=0)
    flip_cam = st.checkbox("Mirror Kamera (Cermin)", value=True)

    st.markdown("---")
    st.subheader("⚙️ Sensitivitas Deteksi")
    conf_threshold = st.slider(
        "Ambang Keyakinan (Confidence):",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Jika probabilitas model di bawah nilai ini, angka dianggap 'Unknown'."
    )

    st.markdown("---")
    st.markdown("### 📖 Panduan Posisi Angka")
    st.markdown("""
    <div class="gesture-guide-pill"><span>✊ <b>Angka 0</b></span><span style="color:#94a3b8;">Tangan Mengepal</span></div>
    <div class="gesture-guide-pill"><span>☝️ <b>Angka 1</b></span><span style="color:#94a3b8;">1 Jari (Telunjuk)</span></div>
    <div class="gesture-guide-pill"><span>✌️ <b>Angka 2</b></span><span style="color:#94a3b8;">2 Jari (V sign)</span></div>
    <div class="gesture-guide-pill"><span>🤟 <b>Angka 3</b></span><span style="color:#94a3b8;">3 Jari</span></div>
    <div class="gesture-guide-pill"><span>🖖 <b>Angka 4</b></span><span style="color:#94a3b8;">4 Jari</span></div>
    <div class="gesture-guide-pill"><span>🖐️ <b>Angka 5</b></span><span style="color:#94a3b8;">5 Jari Terbuka</span></div>
    """, unsafe_allow_html=True)


predictor = get_predictor(confidence_threshold=conf_threshold)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🖐️ Live Deteksi Angka Gestur Tangan (0–5)")
st.caption("Arahkan tangan ke depan webcam. Angka akan langsung terdeteksi dan muncul secara otomatis tanpa harus memencet tombol ambil foto.")

tab_live, tab_eval, tab_upload = st.tabs([
    "🔴 Live Webcam (Otomatis)",
    "📈 Evaluasi Model",
    "🖼️ Uji Gambar"
])


# ---------------------------------------------------------------------------
# TAB 1: Live Real-Time Detection
# ---------------------------------------------------------------------------
with tab_live:
    col_cam, col_info = st.columns([1.6, 1.0], gap="large")

    with col_info:
        st.subheader("🎯 Hasil Angka Terdeteksi")
        number_placeholder = st.empty()
        stats_placeholder = st.empty()
        probs_placeholder = st.empty()

        # Initial state
        number_placeholder.markdown("""
        <div class="hero-badge-container">
            <span class="hero-status-pill pill-gray">Menunggu Tangan</span>
            <div class="hero-number-empty">—</div>
            <div class="hero-label">Tunjukkan Tangan Anda</div>
            <span style="color: #94a3b8; font-size: 0.85rem;">Posisikan tangan di depan webcam</span>
        </div>
        """, unsafe_allow_html=True)

    with col_cam:
        c_btn1, c_btn2, c_spacer = st.columns([1, 1, 1.5])
        if c_btn1.button("▶️ Lanjutkan Live", type="primary", use_container_width=True):
            st.session_state.camera_running = True
            st.rerun()
        if c_btn2.button("⏹️ Jeda / Berhenti", use_container_width=True):
            st.session_state.camera_running = False
            st.rerun()

        frame_placeholder = st.empty()

        if not st.session_state.camera_running:
            frame_placeholder.info("Kamera sedang dijeda. Klik tombol **▶️ Lanjutkan Live** untuk menyalakan kembali deteksi langsung.")
            number_placeholder.markdown("""
            <div class="hero-badge-container">
                <span class="hero-status-pill pill-orange">Kamera Dijeda</span>
                <div class="hero-number-empty">⏸️</div>
                <div class="hero-label">Live Deteksi Nonaktif</div>
                <span style="color: #94a3b8; font-size: 0.85rem;">Klik Lanjutkan Live untuk mulai</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            cap = cv2.VideoCapture(cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                frame_placeholder.error(
                    f"Tidak dapat mengakses kamera index {cam_index}. "
                    "Pastikan webcam terhubung dan tidak sedang digunakan program lain."
                )
            else:
                prev_time = time.time()
                try:
                    while st.session_state.camera_running:
                        ok, frame = cap.read()
                        if not ok:
                            frame_placeholder.warning("Gagal membaca frame kamera.")
                            break

                        if flip_cam:
                            frame = cv2.flip(frame, 1)

                        t0 = time.time()
                        result = predictor.predict_frame(frame, draw_overlay=True)
                        latency_ms = (time.time() - t0) * 1000
                        fps = 1.0 / (time.time() - prev_time + 1e-6)
                        prev_time = time.time()

                        # Update preview frame in placeholder
                        frame_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, use_container_width=True)

                        # Update number card on the right
                        if not result["hand_detected"]:
                            number_placeholder.markdown("""
                            <div class="hero-badge-container">
                                <span class="hero-status-pill pill-gray">Menunggu Tangan</span>
                                <div class="hero-number-empty">—</div>
                                <div class="hero-label">Tunjukkan Tangan Anda</div>
                                <span style="color: #94a3b8; font-size: 0.85rem;">Posisikan jari 0–5 di depan webcam</span>
                            </div>
                            """, unsafe_allow_html=True)

                            stats_placeholder.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Status Kamera</div>
                                <div class="metric-value" style="font-size: 1.3rem;">Live ~ {fps:.0f} FPS</div>
                                <div class="metric-sub">MediaPipe Standby ({latency_ms:.1f} ms)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif result["is_unknown"]:
                            number_placeholder.markdown(f"""
                            <div class="hero-badge-container">
                                <span class="hero-status-pill pill-orange">Ragu-ragu</span>
                                <div class="hero-number-unknown">?</div>
                                <div class="hero-label">Tidak Meyakinkan</div>
                                <span style="color: #fbbf24; font-size: 0.85rem;">Keyakinan {result['confidence_percentage']}% di bawah ambang {int(conf_threshold*100)}%</span>
                            </div>
                            """, unsafe_allow_html=True)

                            stats_placeholder.markdown(f"""
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                                <div class="metric-card">
                                    <div class="metric-title">Kecepatan</div>
                                    <div class="metric-value">{fps:.0f} <span style="font-size: 0.9rem;">FPS</span></div>
                                    <div class="metric-sub">{latency_ms:.1f} ms</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">Keyakinan</div>
                                    <div class="metric-value" style="color: #fbbf24;">{result['confidence_percentage']:.0f}%</div>
                                    <div class="metric-sub">Unknown (&lt;{int(conf_threshold*100)}%)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            detected_num = result["predicted_label"]
                            conf_val = result["confidence_percentage"]

                            number_placeholder.markdown(f"""
                            <div class="hero-badge-container">
                                <span class="hero-status-pill pill-green">✓ Terdeteksi Langsung</span>
                                <div class="hero-number">{detected_num}</div>
                                <div class="hero-label">Angka {detected_num}</div>
                                <span style="color: #38bdf8; font-size: 1rem; font-weight: 700;">Keyakinan: {conf_val}%</span>
                            </div>
                            """, unsafe_allow_html=True)

                            stats_placeholder.markdown(f"""
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                                <div class="metric-card">
                                    <div class="metric-title">Kecepatan</div>
                                    <div class="metric-value">{fps:.0f} <span style="font-size: 0.9rem;">FPS</span></div>
                                    <div class="metric-sub">{latency_ms:.1f} ms</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">Keyakinan</div>
                                    <div class="metric-value" style="color: #38bdf8;">{conf_val:.0f}%</div>
                                    <div class="metric-sub">Tinggi</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        time.sleep(0.015)
                finally:
                    cap.release()


# ---------------------------------------------------------------------------
# TAB 2: Model Evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("📈 Laporan Evaluasi Model Machine Learning")
    metrics_file = Path("reports/metrics.json")
    cm_img_file = Path("reports/confusion_matrix.png")
    report_txt_file = Path("reports/classification_report.txt")

    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Akurasi Testing", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("Macro F1-Score", f"{metrics['f1_macro']*100:.2f}%")
        m3.metric("Macro Precision", f"{metrics['precision_macro']*100:.2f}%")
        m4.metric("Macro Recall", f"{metrics['recall_macro']*100:.2f}%")

        st.markdown("---")
        col_cm, col_rep = st.columns([1.1, 1.0], gap="large")

        with col_cm:
            st.markdown("#### 🧩 Confusion Matrix")
            if cm_img_file.exists():
                st.image(str(cm_img_file), use_container_width=True)

        with col_rep:
            st.markdown("#### 📊 Performa Per Angka (0–5)")
            per_class_data = [
                {
                    "Angka": f"Angka {lbl}",
                    "Precision": f"{d['precision']*100:.1f}%",
                    "Recall": f"{d['recall']*100:.1f}%",
                    "F1-Score": f"{d['f1_score']*100:.1f}%",
                    "Support": d["support"]
                }
                for lbl, d in metrics["per_class"].items()
            ]
            st.dataframe(pd.DataFrame(per_class_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 3: Test Image Upload
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("🖼️ Uji dari File Gambar")
    up_file = st.file_uploader("Unggah foto tangan:", type=["jpg", "jpeg", "png"])
    if up_file:
        img_pil = Image.open(up_file).convert("RGB")
        cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        res = predictor.predict_frame(cv_img, draw_overlay=True)

        col_u1, col_u2 = st.columns([1.5, 1.0])
        with col_u1:
            st.image(cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_u2:
            if not res["hand_detected"]:
                st.error("Tangan tidak terdeteksi.")
            elif res["is_unknown"]:
                st.warning(f"Unknown (Keyakinan {res['confidence_percentage']}%)")
            else:
                st.success(f"Angka Terdeteksi: **{res['predicted_label']}** ({res['confidence_percentage']}%)")
