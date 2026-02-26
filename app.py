from flask import Flask, request, jsonify
import numpy as np
import librosa
import pickle
import tensorflow as tf
import os
import tempfile
import redis
import hashlib
import json

app = Flask(__name__)

# =============================
# LOAD MODEL + PREPROCESSING
# =============================

model = tf.keras.models.load_model("cnn_ser_model.keras")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

mean, std = np.load("normalization_values.npy")

# =============================
# CONNECT TO REDIS
# =============================

cache = redis.Redis(host='localhost', port=6379, db=0)

# =============================
# AUDIO PREPROCESSING
# =============================

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        hop_length=512
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    max_len = 128
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]

    log_mel = log_mel[..., np.newaxis]
    log_mel = (log_mel - mean) / (std + 1e-6)
    log_mel = np.expand_dims(log_mel, axis=0)

    return log_mel

# =============================
# ROUTES
# =============================

@app.route("/")
def home():
    return "Speech Emotion Recognition API with Redis Cache is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Read file bytes for hashing
        file_bytes = file.read()

        # Generate unique hash for caching
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # =============================
        # CHECK REDIS CACHE
        # =============================

        cached_result = cache.get(file_hash)

        if cached_result:
            print("Returned from Redis cache")
            return jsonify(json.loads(cached_result))

        # =============================
        # SAVE TEMP FILE
        # =============================

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        # =============================
        # PREPROCESS + PREDICT
        # =============================

        processed = preprocess_audio(temp_path)

        prediction = model.predict(processed)[0]

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        emotion = le.inverse_transform([predicted_class])[0]

        response_data = {
            "emotion": emotion,
            "confidence": confidence,
            "all_probabilities": {
                le.inverse_transform([i])[0]: float(prediction[i])
                for i in range(len(prediction))
            }
        }

        # =============================
        # STORE IN REDIS (1 hour expiry)
        # =============================

        cache.setex(
            file_hash,
            3600,  # seconds
            json.dumps(response_data)
        )

        return jsonify(response_data)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

# =============================
# RUN APP
# =============================

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)