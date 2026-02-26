import os
import sys
import librosa
import numpy as np
from pyspark.sql import SparkSession

# Force Spark to use current Python interpreter
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# =============================
# Start Spark Session
# =============================

spark = SparkSession.builder \
    .appName("SpeechEmotionBatchProcessing") \
    .getOrCreate()

print("Spark Session Created")

# =============================
# Dataset Path
# =============================

DATASET_PATH = "dataset"   # adjust if needed

# Collect all audio file paths
audio_paths = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            audio_paths.append(os.path.join(root, file))

print(f"Total files found: {len(audio_paths)}")

# Convert to Spark DataFrame
df = spark.createDataFrame([(path,) for path in audio_paths], ["path"])

# =============================
# Feature Extraction Function
# =============================

def extract_features(path):
    try:
        audio, sr = librosa.load(path, sr=22050)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return float(np.mean(log_mel))
    except:
        return None

# Register as UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

extract_udf = udf(extract_features, DoubleType())

# Apply feature extraction in distributed way
df_features = df.withColumn("logmel_mean", extract_udf(df["path"]))

# Show result
df_features.show(5)

print("Spark batch processing complete.")

spark.stop()