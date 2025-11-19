import streamlit as st
import cv2
import time
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Realtime Detection", layout="wide")

st.title("📷 YOLO Realtime Object Detection with Streamlit")

# Provide a helpful default model path and a labeled input field
model_path = st.text_input("Model path", value="./runs/segment/train8/weights/best.pt")

@st.cache_resource
def load_model(path):
    return YOLO(path)

# persist model and running state across reruns
if "model" not in st.session_state:
    st.session_state.model = None
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Load model"):
        if model_path:
            try:
                st.session_state.model = load_model(model_path)
                st.success(f"Model loaded from {model_path}")
            except Exception as e:
                st.session_state.model = None
                st.error(f"Error loading model: {e}")
        else:
            st.warning("Please provide a model path before loading.")

with col2:
    if st.button("Start Camera"):
        if st.session_state.model is None:
            st.warning("Load a model first.")
        else:
            st.session_state.running = True
    if st.button("Stop Camera"):
        st.session_state.running = False

# Confidence threshold slider and option to show visual bars
# Default to 0.8 so only detections >=80% are shown by default
min_conf = st.slider("Confidence threshold", 0.0, 1.0, 0.8, 0.01)
show_conf_bars = st.checkbox("Show confidence bars on frame", value=True)

model = st.session_state.model
FRAME_WINDOW = st.image([])
conf_progress = st.progress(0.0)

if st.session_state.running:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Failed to open webcam. Check permissions and device index.")
        st.session_state.running = False
    else:
        try:
            while st.session_state.running:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break

                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run detection (guard in case model becomes None)
                try:
                    results = model(frame, stream=True)
                except Exception as e:
                    st.error(f"Model inference error: {e}")
                    break

                # Draw results (filter by confidence threshold)
                max_conf = 0.0
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if conf < min_conf:
                            continue
                        max_conf = max(max_conf, conf)
                        label = f"{model.names[cls]} {conf:.2f}"

                        # Bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Draw a confidence bar under the box if enabled
                        if show_conf_bars:
                            bar_height = 6
                            full_w = max(1, x2 - x1)
                            filled_w = int(full_w * conf)
                            bar_y1 = min(frame.shape[0] - 1, y2 + 6)
                            bar_y2 = min(frame.shape[0], y2 + 6 + bar_height)
                            # background (dark gray)
                            cv2.rectangle(frame, (x1, bar_y1), (x1 + full_w, bar_y2), (60, 60, 60), -1)
                            # filled portion (green)
                            cv2.rectangle(frame, (x1, bar_y1), (x1 + filled_w, bar_y2), (0, 220, 0), -1)

                # update confidence progress indicator
                try:
                    conf_progress.progress(max_conf)
                except Exception:
                    pass

                # Show frame
                FRAME_WINDOW.image(frame)
                time.sleep(0.03)
        finally:
            camera.release()