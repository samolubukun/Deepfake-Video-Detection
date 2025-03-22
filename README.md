# Deepfake Video Detection using CNN and LSTM

This repository provides a deep learning-based solution for detecting deepfake videos using a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The system analyzes uploaded videos and predicts whether the video is real or deepfake using a trained deep learning model.

![image](https://github.com/user-attachments/assets/550119c7-cbd5-4160-b8e5-0dc86bc79115)

## Features
- **Streamlit Web App:** A user-friendly interface for uploading and analyzing videos.
- **Pretrained InceptionV3 Model:** Used for feature extraction from video frames.
- **LSTM-Based Classification** Sequences of extracted features are processed through an LSTM network for classification.
- **Efficient Face Detection:** MTCNN is used to detect faces in videos, ensuring accurate feature extraction.
- **High Accuracy:** The trained model achieves an overall accuracy of 81.25% on the test dataset.

## Model Summary
The gru LSTM model was trained to classify sequences of videos as deepfake or real:
- **Architecture:** Feature extractor (InceptionV3, outputting 2048-dimensional feature vectors). LSTM network with two GRU layers (16 and 8 units respectively).
- **Performance Metrics:**
  - Accuracy: 81.27%
  - Precision: 87.0%
  - Recall: 86.0%
