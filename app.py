import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import IMG_SIZE
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model("asl_model.h5")

# Map numeric class back to letter (same order as training)
class_labels = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# Streamlit UI
st.title("🤟 Real-Time ASL Sign Recognition")
st.markdown("Show a hand sign to the webcam and get the predicted letter!")
cap = None
# Release camera
if cap:
    cap.release()
    st.sidebar.title("Instructions")
    st.sidebar.write("""
    - Click 'Start Camera'
    - Show a hand sign from A-Z
    - Make sure lighting is good and hand is centered
    """)

# Webcam feed
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])


if run:
    cap = cv2.VideoCapture(0)

while run and cap is not None and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture frame")
        break

    # Convert to grayscale, resize, normalize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm_img = resized / 255.0
    input_img = norm_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    prediction = model.predict(input_img)
    # st.write("Prediction raw:", prediction)
    # st.write("Prediction shape:", prediction.shape)
    # st.write("Class labels:", class_labels)
    confidence = np.max(prediction)
    st.write(f"Confidence: {confidence:.2f}")

    predicted_class = np.argmax(prediction)
    if 0 <= predicted_class < len(class_labels):
        predicted_label = class_labels[predicted_class]
        st.success(f"Prediction: {predicted_label}")
        # Show label on frame
        cv2.putText(frame, f'Prediction: {predicted_label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        st.error("Prediction out of range. Try again with a clearer image.")
        st.write("Predicted class index:", predicted_class)

    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)