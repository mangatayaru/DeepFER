import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw
import cv2

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "deepfer_best_model.keras"
META_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="DeepFER | Facial Expression Classifier", page_icon="🙂", layout="centered")

@st.cache_resource
def load_artifacts():
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, metadata

model, metadata = load_artifacts()
CLASS_NAMES = metadata["class_names"]

st.title("DeepFER")
st.subheader("Facial Expression Image Classifier")
st.caption("Experimental computer-vision demonstration. It estimates visible expression categories; it does not determine a person's actual feelings, intentions, personality, or mental health.")

with st.expander("About this model and its limitations"):
    st.write(
        "The selected model is a fine-tuned MobileNetV2 trained on 48×48 grayscale face images, "
        "then evaluated on a held-out test split. Performance varies by class and image quality. "
        "Do not use this output for hiring, education, policing, healthcare, identity verification, "
        "or other consequential decisions about people."
    )
    st.write("Selected model:", metadata.get("selected_model", "Unknown"))
    st.write("Held-out test accuracy:", f'{metadata.get("test_accuracy", 0):.1%}')
    st.write("Held-out test macro F1:", f'{metadata.get("test_macro_f1", 0):.3f}')

def detect_and_crop_face(pil_image):
    """Return largest detected face crop; if none found, return original with a notice."""
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return pil_image.convert("RGB"), False
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    # Add a modest margin while staying within image boundaries.
    margin = int(0.15 * max(w, h))
    x1, y1 = max(0, x-margin), max(0, y-margin)
    x2, y2 = min(rgb.shape[1], x+w+margin), min(rgb.shape[0], y+h+margin)
    return Image.fromarray(rgb[y1:y2, x1:x2]), True

def preprocess(pil_image):
    # Training pipeline: grayscale 48x48 -> RGB -> resize 96x96.
    gray = ImageOps.grayscale(pil_image)
    rgb = Image.merge("RGB", (gray, gray, gray))
    rgb = rgb.resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict(pil_image):
    batch = preprocess(pil_image)
    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return [(CLASS_NAMES[i], float(probabilities[i])) for i in order]

st.markdown("### Provide an image")
input_method = st.radio("Input method", ["Upload an image", "Use camera"], horizontal=True)
uploaded = None
if input_method == "Upload an image":
    uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
else:
    uploaded = st.camera_input("Take a photo")

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        face_crop, face_found = detect_and_crop_face(image)
        st.image(image, caption="Original input", use_container_width=True)
        if face_found:
            st.success("A face was detected. The largest detected face crop is used for prediction.")
            st.image(face_crop, caption="Face crop used by the model", width=240)
        else:
            st.warning("No face was detected. The full image will be classified; results may be unreliable. Try a clear, front-facing face image.")

        if st.button("Classify expression", type="primary"):
            results = predict(face_crop)
            label, score = results[0]
            st.markdown("### Prediction")
            st.metric("Most likely visible expression category", label.title(), f"{score:.1%} model score")
            st.caption("The score is the model's softmax output, not a calibrated probability that the person feels this emotion.")
            st.markdown("### Class scores")
            for class_name, value in results:
                st.write(f"**{class_name.title()}** — {value:.1%}")
                st.progress(value)
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption("DeepFER academic project • Use only with appropriate consent and respect for privacy.")
