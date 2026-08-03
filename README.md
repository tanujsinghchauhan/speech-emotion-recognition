# Speech Emotion Recognition System

## Project Overview

This project implements a **Real-Time Speech Emotion Recognition (SER)**
system using Deep Learning and Distributed Processing.

The system is capable of: - Extracting Log-Mel Spectrogram features from
audio - Training a Convolutional Neural Network (CNN) - Serving
predictions via a Flask API - Providing an interactive Streamlit web
interface - Caching predictions using Redis for performance
optimization - Performing batch feature extraction using Apache Spark -
Visualizing analytics using Tableau

---

## System Architecture

User → Streamlit → Flask API → Redis Cache → CNN Model → Response\
Dataset → Spark Batch Processing → Feature Data → TensorFlow Training

---

## Project Folder Structure

    speech-emotion-recognition/
    │
    ├── app.py                      # Flask backend API
    ├── streamlit_app.py            # Streamlit frontend UI
    ├── spark_batch_processing.py   # Spark distributed feature extraction
    ├── cnn_ser_model.keras         # Trained CNN model
    ├── label_encoder.pkl           # Label encoder for emotions
    ├── normalization_values.npy    # Mean & Std for normalization
    ├── requirements.txt            # Project dependencies
    ├── analytics_data/             # Generated analytics CSV files
    ├── notebooks/                  # Jupyter notebooks (training & experiments)
    └── dataset/                    # Audio dataset (RAVDESS)

---

## Model Details

- Feature Type: Log-Mel Spectrogram (128x128)
- Model: Convolutional Neural Network (CNN)
- Classes:
  - Angry
  - Calm
  - Disgust
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised
- Framework: TensorFlow / Keras

---

## Installation Guide

### Clone the Repository

    git clone https://github.com/tanujsinghchauhan/speech-emotion-recognition
    cd speech-emotion-recognition

---

### Create Virtual Environment (Recommended)

    python -m venv ser_env
    ser_env\Scripts\activate   # Windows

---

### Install Dependencies

    pip install -r requirements.txt

---

## Dataset Setup

This project uses the **RAVDESS dataset**.

1.  Download the dataset.

```
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
```

2.  Extract it into the project root directory.
3.  Ensure the folder structure is:

```
    speech-emotion-recognition/
    └── dataset/
        ├── Actor_01/
        ├── Actor_02/
        └── ...
```

---

## How to Run the System

### Step 1: Start Redis Server

    redis-server

---

### Step 2: Start Flask Backend

    python app.py

Backend runs at:

    http://127.0.0.1:5000

---

### Step 3: Start Streamlit Frontend

    streamlit run streamlit_app.py

Frontend runs at:

    http://localhost:8501

---

## How the System Works

1.  User uploads or records audio via Streamlit.
2.  Streamlit sends audio to Flask API.
3.  Flask:
    - Generates file hash
    - Checks Redis cache
    - If cached → returns result instantly
    - If not cached → preprocesses audio → CNN prediction → stores
      result in Redis
4.  Response returned to Streamlit.
5.  Streamlit displays:
    - Predicted emotion
    - Confidence score
    - Probability distribution
    - Spectrogram preview

---

## Redis Caching

Redis is used as an in-memory cache to:

- Avoid repeated CNN inference
- Reduce latency
- Improve scalability

Cached predictions expire automatically after a fixed time.

---

## Apache Spark Integration

Spark is used for distributed batch feature extraction:

- Loads all dataset audio paths
- Extracts log-mel features in parallel
- Prepares data for training or analytics

---

## Analytics

Model performance metrics are exported to CSV files and visualized in
Tableau, including:

- Accuracy comparison
- Confusion matrices
- Precision / Recall / F1-score
- Model comparison (CNN vs SVM)

---

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- Librosa
- NumPy & Pandas
- Scikit-learn
- Flask
- Streamlit
- Redis
- Apache Spark
- Tableau

---

## Future Improvements

- Deploy on cloud (AWS / Azure)
- Add real-time streaming with Kafka
- Implement hybrid CNN-LSTM architecture
- Add user authentication
- Dockerize the entire system

---

## Live Demo

Try it here: **[speech-emotion-recognition.streamlit.app](https://speech-emotion-recognition-diuq5ftayfobwhcbhypv2g.streamlit.app/)**

- **Note on deployment:** The architecture above (Flask API + Redis caching + Spark batch
- processing) reflects the intended production design and is fully implemented in `app.py`
- and `spark_batch_processing.py`. For the free-tier live demo, the Flask and Redis layers
- are bypassed — `streamlit_app.py` loads the CNN model and runs inference directly in-process,
- since Streamlit Community Cloud only supports a single Python service. The prediction logic
- itself (preprocessing, model, feature extraction) is identical in both versions.
- To run the full multi-service architecture locally, follow the installation guide.

---

## Author's Note

Developed as part of a full-stack Machine Learning system demonstrating
model training, deployment, caching, distributed processing, and
analytics integration.

---

## License

This project is for educational and research purposes.
