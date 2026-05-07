import streamlit as st
import cv2
import numpy as np
import json
import tensorflow as tf
import mediapipe as mp
from collections import deque
from PIL import Image
import tempfile
import os

# ── Page Config ──
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🤟",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .letter-display {
        font-size: 80px;
        font-weight: bold;
        line-height: 1;
    }
    .confidence-text {
        font-size: 20px;
        margin-top: 10px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .word-buffer {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #cdd6f4;
        font-size: 28px;
        font-family: monospace;
        letter-spacing: 4px;
        min-height: 70px;
    }
    .top3-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
@st.cache_resource
def load_model():
    import urllib.request
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model = tf.keras.models.load_model("models/asl_landmark_model.h5")

    with open("models/label_map.json") as f:
        label_map = json.load(f)

    idx_to_label = {v: k for k, v in label_map.items()}

    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            model_path
        )

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    hands = vision.HandLandmarker.create_from_options(options)

    return model, idx_to_label, hands

# Download hand landmarker model
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
hands = vision.HandLandmarker.create_from_options(options)
    return model, idx_to_label, hands

model, idx_to_label, hands = load_model()

def normalize(landmarks):
    pts = np.array(landmarks, dtype=np.float32)
    pts = pts - pts[0]
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0:
        pts = pts / d
    return pts.flatten()

def predict_image(img_bgr):
    import mediapipe as mp
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hands.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None, "No hand detected"

    hand = result.hand_landmarks[0]
    h, w, _ = img_bgr.shape
    pts = [(int(lm.x*w), int(lm.y*h), lm.z) for lm in hand]
    feat = normalize(pts).reshape(1, -1)
    probs = model.predict(feat, verbose=0)[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    top1_conf = probs[top3_idx[0]]
    top1_label = idx_to_label[top3_idx[0]]
    top3 = [(idx_to_label[i], float(probs[i])) for i in top3_idx]

    for lm in hand:
        cx, cy = int(lm.x*w), int(lm.y*h)
        cv2.circle(img_bgr, (cx, cy), 5, (0, 255, 0), -1)

    return top1_label, float(top1_conf), top3, None
# ── Header ──
st.markdown('<div class="main-header">🤟 ASL Recognition System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">American Sign Language Alphabet Recognition using Deep Learning</div>',
            unsafe_allow_html=True)

# ── Stats Row ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", "99.14%", "↑ 9.35% from baseline")
with col2:
    st.metric("Classes", "29", "A-Z + del/space/nothing")
with col3:
    st.metric("Training Samples", "~60K", "Filtered from 87K")
with col4:
    st.metric("Architecture", "MLP", "63-dim landmark vectors")

st.divider()

# ── Tabs ──
tab1, tab2, tab3 = st.tabs([
    "📸 Image Upload",
    "📝 Word Builder",
    "📊 Model Info"
])

# ══════════════════════════════════════
# TAB 1 — Image Upload
# ══════════════════════════════════════
with tab1:
    st.subheader("Upload a hand sign image")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a hand showing an ASL letter"
        )

        if uploaded_file:
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()), dtype=np.uint8
            )
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_display = img.copy()

            st.image(
                cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB),
                caption="Uploaded Image",
                use_column_width=True
            )

    with col_result:
        if uploaded_file is not None:
            with st.spinner("Analyzing hand sign..."):
                label, conf, top3, error = predict_image(img)

            if error:
                st.error(f"❌ {error}")
                st.info("💡 Tips: Ensure hand is clearly visible, good lighting, plain background")
            else:
                # Main result
                status = "Confident ✅" if conf >= 0.85 else "Uncertain ⚠️"
                display_label = label if conf >= 0.85 else "?"

                st.markdown(f"""
                <div class="result-box">
                    <div class="letter-display">{display_label}</div>
                    <div class="confidence-text">
                        Confidence: {conf*100:.1f}% — {status}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                st.progress(conf, text=f"Confidence: {conf*100:.1f}%")

                # Top 3
                st.markdown("#### 🏆 Top 3 Predictions")
                for i, (lbl, prob) in enumerate(top3):
                    medal = ["🥇", "🥈", "🥉"][i]
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(prob, text=f"{medal} {lbl}")
                    with col_b:
                        st.write(f"**{prob*100:.1f}%**")

                # Landmark image
                st.markdown("#### 🖐 Detected Landmarks")
                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    caption="Hand landmarks detected",
                    use_column_width=True
                )

# ══════════════════════════════════════
# TAB 2 — Word Builder
# ══════════════════════════════════════
with tab2:
    st.subheader("Build words by uploading signs one at a time")
    st.info("Upload hand signs one by one to spell out words!")

    # Initialize word buffer in session state
    if "word" not in st.session_state:
        st.session_state.word = ""
    if "history" not in st.session_state:
        st.session_state.history = []

    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        word_upload = st.file_uploader(
            "Upload next letter",
            type=["jpg", "jpeg", "png"],
            key="word_uploader"
        )

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("➕ Add Space"):
                st.session_state.word += " "
                st.session_state.history.append("SPACE")
        with col_btn2:
            if st.button("⬅️ Backspace"):
                st.session_state.word = st.session_state.word[:-1]
                if st.session_state.history:
                    st.session_state.history.pop()
        with col_btn3:
            if st.button("🗑️ Clear"):
                st.session_state.word = ""
                st.session_state.history = []

        if word_upload:
            file_bytes = np.asarray(
                bytearray(word_upload.read()), dtype=np.uint8
            )
            img_w = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Detecting..."):
                label, conf, top3, error = predict_image(img_w)

            if error:
                st.error(f"❌ {error}")
            else:
                if conf >= 0.85:
                    if label not in ["nothing", "space", "del"]:
                        st.session_state.word += label
                        st.session_state.history.append(label)
                        st.success(f"Added: **{label}** ({conf*100:.1f}%)")
                    elif label == "del":
                        st.session_state.word = st.session_state.word[:-1]
                        st.warning("Deleted last letter")
                    elif label == "space":
                        st.session_state.word += " "
                        st.info("Added space")
                else:
                    st.warning(f"Low confidence ({conf*100:.1f}%) — not added")

            st.image(
                cv2.cvtColor(img_w, cv2.COLOR_BGR2RGB),
                caption=f"Detected: {label} ({conf*100:.1f}%)" if not error else "No hand",
                use_column_width=True
            )

    with col_w2:
        st.markdown("#### 📝 Current Word")
        word_display = st.session_state.word if st.session_state.word else "..."
        st.markdown(f'<div class="word-buffer">{word_display}</div>',
                    unsafe_allow_html=True)

        st.markdown("#### 📜 Letter History")
        if st.session_state.history:
            history_str = " → ".join(st.session_state.history[-10:])
            st.code(history_str)
        else:
            st.code("No letters added yet")

        if st.session_state.word:
            st.markdown("#### 💾 Export")
            st.download_button(
                "Download Word",
                data=st.session_state.word,
                file_name="asl_word.txt",
                mime="text/plain"
            )

# ══════════════════════════════════════
# TAB 3 — Model Info
# ══════════════════════════════════════
with tab3:
    st.subheader("Model Architecture & Training Details")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### 🧠 Architecture")
        st.code("""
Input: 63-dim landmark vector
    ↓
Dense(512) + BatchNorm + Dropout(0.4)
    ↓
Dense(256) + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU
    ↓
Dense(29) + Softmax
        """)

        st.markdown("#### 📦 Tech Stack")
        tech = {
            "Framework": "TensorFlow 2.13",
            "Hand Detection": "MediaPipe 0.10.14",
            "Computer Vision": "OpenCV 4.8",
            "Language": "Python 3.10",
            "UI": "Streamlit"
        }
        for k, v in tech.items():
            st.markdown(f"- **{k}:** {v}")

    with col_m2:
        st.markdown("#### 📈 Training Results")
        results = {
            "Run 1 (500/class)": "89.79%",
            "Run 2 (3000/class, unfiltered)": "75.94%",
            "Run 3 (3000/class, filtered)": "99.14% ✅"
        }
        for run, acc in results.items():
            st.markdown(f"- **{run}:** {acc}")

        st.markdown("#### 🔍 Key Findings")
        st.markdown("""
- **M vs N** most confused pair (visually similar in ASL)
- **B, C, L, Y** — perfect classification
- Filtering bad MediaPipe detections improved accuracy by **23%**
- Bigger model (512 neurons) helped with 60K samples
        """)

        st.markdown("#### 🖼️ Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_column_width=True)
        else:
            st.info("Run confusion_matrix.py to generate this")

# ── Footer ──
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 13px;'>
    Built with TensorFlow · MediaPipe · Streamlit<br>
    ASL Recognition System — BTech CV Project
</div>
""", unsafe_allow_html=True)
