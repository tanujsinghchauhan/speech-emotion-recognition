import streamlit as st
import requests
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
from audiorecorder import audiorecorder

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("Speech Emotion Recognition")
st.markdown("Upload or record your voice to detect emotional state.")

st.divider()

# =============================
# INPUT SECTION
# =============================

uploaded_file = st.file_uploader("📁 Upload a WAV file", type=["wav"])

st.markdown("### Or Record Your Voice")
audio_recording = audiorecorder("Click to record", "Recording...")

audio_bytes = None

if len(audio_recording) > 0:
    recorded_bytes = audio_recording.export().read()
    st.audio(recorded_bytes, format="audio/wav")
    audio_bytes = recorded_bytes

elif uploaded_file is not None:
    uploaded_bytes = uploaded_file.read()
    st.audio(uploaded_bytes, format="audio/wav")
    audio_bytes = uploaded_bytes

st.divider()

# =============================
# ANALYZE BUTTON
# =============================

if st.button("Analyze Emotion", use_container_width=True):

    if audio_bytes is None:
        st.warning("Please upload or record audio first.")
    else:
        with st.spinner("Analyzing emotion..."):

            # Save audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            # commented out for hosting simplicity
            # try:
            #     with open(temp_path, "rb") as f:
            #         response = requests.post(
            #             "http://127.0.0.1:5000/predict",
            #             files={"file": f}
            #         )

            #     if response.status_code == 200:
            #         result = response.json()

            #         emotion = result["emotion"]
            #         confidence = result["confidence"]
            #         probs = result["all_probabilities"]

            #         # =============================
            #         # DISPLAY RESULTS
            #         # =============================

            #         st.success(f"Predicted Emotion: **{emotion.upper()}**")
            #         st.write(f"Confidence Score: **{confidence:.4f}**")

            #         if confidence > 0.8:
            #             st.info("High confidence prediction.")
            #         elif confidence > 0.5:
            #             st.warning("Moderate confidence prediction.")
            #         else:
            #             st.error("Low confidence prediction.")

            #         # =============================
            #         # PROBABILITY BAR CHART
            #         # =============================

            #         st.markdown("### Emotion Probability Distribution")

            #         emotions = list(probs.keys())
            #         probabilities = list(probs.values())

            #         fig, ax = plt.subplots()
            #         ax.barh(emotions, probabilities)
            #         ax.set_xlim([0, 1])
            #         ax.set_xlabel("Probability")
            #         st.pyplot(fig)

            #         # =============================
            #         # SPECTROGRAM PREVIEW
            #         # =============================

            #         st.markdown("### Spectrogram Preview")

            #         audio, sr = librosa.load(temp_path, sr=22050)
            #         mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            #         log_mel = librosa.power_to_db(mel, ref=np.max)

            #         fig2, ax2 = plt.subplots()
            #         ax2.imshow(log_mel, aspect='auto', origin='lower')
            #         ax2.set_title("Log-Mel Spectrogram")
            #         st.pyplot(fig2)

            #     else:
            #         st.error("Prediction failed.")

            # except Exception as e:
            #     st.error(f"Connection error: {e}")
            try:
                import numpy as np
                import pickle
                import tensorflow as tf

                @st.cache_resource
                def load_model_and_preprocessing():
                    model = tf.keras.models.load_model("cnn_ser_model.keras")
                    with open("label_encoder.pkl", "rb") as f:
                        le = pickle.load(f)
                    mean, std = np.load("normalization_values.npy")
                    return model, le, mean, std

                model, le, mean, std = load_model_and_preprocessing()

                def preprocess_audio(file_path):
                    audio, sr = librosa.load(file_path, sr=22050)
                    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512)
                    log_mel = librosa.power_to_db(mel, ref=np.max)
                    max_len = 128
                    if log_mel.shape[1] < max_len:
                        pad_width = max_len - log_mel.shape[1]
                        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
                    else:
                        log_mel = log_mel[:, :max_len]
                        log_mel = log_mel[..., np.newaxis]
                        log_mel = (log_mel - mean) / (std + 1e-6)
                        return np.expand_dims(log_mel, axis=0)

                processed = preprocess_audio(temp_path)
                prediction = model.predict(processed)[0]

                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                emotion = le.inverse_transform([predicted_class])[0]
                probs = {le.inverse_transform([i])[0]: float(prediction[i]) for i in range(len(prediction))}

    # =============================
    # DISPLAY RESULTS
    # =============================

                st.success(f"Predicted Emotion: **{emotion.upper()}**")
                st.write(f"Confidence Score: **{confidence:.4f}**")

                if confidence > 0.8:
                    st.info("High confidence prediction.")
                elif confidence > 0.5:
                    st.warning("Moderate confidence prediction.")
                else:
                    st.error("Low confidence prediction.")

                st.markdown("### Emotion Probability Distribution")
                emotions = list(probs.keys())
                probabilities = list(probs.values())

                fig, ax = plt.subplots()
                ax.barh(emotions, probabilities)
                ax.set_xlim([0, 1])
                ax.set_xlabel("Probability")
                st.pyplot(fig)

                st.markdown("### Spectrogram Preview")
                audio, sr = librosa.load(temp_path, sr=22050)
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
                log_mel_display = librosa.power_to_db(mel, ref=np.max)

                fig2, ax2 = plt.subplots()
                ax2.imshow(log_mel_display, aspect='auto', origin='lower')
                ax2.set_title("Log-Mel Spectrogram")
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Prediction error: {e}")

            finally:
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.divider()

# =============================
# MODEL INFO SECTION
# =============================

with st.expander("ℹ Model Information"):
    st.markdown("""
    - Model Type: Convolutional Neural Network (CNN)
    - Feature Extraction: Log-Mel Spectrogram (128x128)
    - Dataset: RAVDESS
    - Classes: Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised
    - Backend: Flask API
    """)