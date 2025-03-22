import streamlit as st
import numpy as np
import cv2 as cv
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
s
# Load pre-trained model and vocabulary
deepfake_model = load_model('video_classifier_full_model.h5')
vocabulary2 = np.load('label_processor_vocabulary.npy', allow_pickle=True)
label_processor2 = keras.layers.StringLookup(num_oov_indices=0, vocabulary=vocabulary2.tolist())

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Face detector
detector = MTCNN()

# Prepare video frames for feature extraction
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask

# Load video frames
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE), skip_frames=2):
    cap = cv.VideoCapture(path)
    frames = []
    frame_count = 0
    previous_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frame, previous_box = get_face_region_first_frame(frame, previous_box)
            if frame is not None:
                frame = cv.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

            if len(frames) == max_frames:
                break
        frame_count += 1

    while len(frames) < max_frames and frames:
        frames.append(frames[-1])

    cap.release()
    return np.array(frames)

# Extract face region
def get_face_region_first_frame(frame, previous_box=None):
    if previous_box is None:
        detections = detector.detect_faces(frame)
        if detections:
            x, y, width, height = detections[0]['box']
            previous_box = (x, y, width, height)
        else:
            return None, None
    else:
        x, y, width, height = previous_box

    face_region = frame[y:y+height, x:x+width]
    return face_region, previous_box

# Sequence prediction
def sequence_prediction(video_path):
    class_vocab = label_processor2.get_vocabulary()
    frames = load_video(video_path)
    if len(frames) == 0:
        st.error("Could not process video. Please try another file.")
        return None

    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = deepfake_model.predict([frame_features, frame_mask])[0]
    
    predictions = {class_vocab[i]: probabilities[i] * 100 for i in np.argsort(probabilities)[::-1]}
    return predictions

# Streamlit App
st.title("Deepfake Video Detection")
st.write("Upload a video to analyze if it is real or fake.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")
    st.write("Analyzing the video...")

    # Initialize progress bar
    progress_bar = st.progress(0)
    
    # Step 1: Load and preprocess video
    progress_bar.progress(20)
    frames = load_video("temp_video.mp4")
    if len(frames) == 0:
        st.error("Could not process video. Please try another file.")
    else:
        progress_bar.progress(50)

        # Step 2: Prepare frames for prediction
        frame_features, frame_mask = prepare_single_video(frames)
        progress_bar.progress(75)

        # Step 3: Perform prediction
        probabilities = deepfake_model.predict([frame_features, frame_mask])[0]
        progress_bar.progress(100)
        
        predictions = {label_processor2.get_vocabulary()[i]: probabilities[i] * 100 for i in np.argsort(probabilities)[::-1]}

        # Display results
        if predictions:
            highest_label = max(predictions, key=predictions.get)
            highest_prob = predictions[highest_label]

            if highest_label.lower() == "real":
                st.success(f"The video is real with a confidence of {highest_prob:.2f}%.")
            elif highest_label.lower() == "fake":
                st.error(f"This video is a deepfake with a confidence of {highest_prob:.2f}%.")
            else:
                st.warning(f"Uncertain prediction: {highest_label} with {highest_prob:.2f}% confidence.")
