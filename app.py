import streamlit as st
import joblib
import librosa
import numpy as np
import tempfile
import os

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Voice Emotion Detection",
    page_icon="🎤",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("emotion_model.pkl")

# =========================
# EMOTION LABELS
# =========================

emotion_labels = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fear",
    6: "Disgust",
    7: "Surprised"
}

# =========================
# TITLE
# =========================

st.title("🎤 AI Voice Emotion Detection")

st.write(
    "Upload a voice recording and let the machine learning model predict the emotion."
)

st.divider()

# =========================
# AUDIO UPLOAD
# =========================

audio_file = st.file_uploader(
    "🎵 Upload your voice recording",
    type=["wav", "mp3", "ogg", "flac"]
)

# =========================
# PREDICTION
# =========================

if audio_file is not None:

    st.audio(
        audio_file,
        format="audio/wav"
    )

    if st.button(
        "🤖 Predict Emotion",
        use_container_width=True
    ):

        try:

            # Save uploaded file temporarily

            suffix = os.path.splitext(
                audio_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    audio_file.getbuffer()
                )

                temp_path = temp_file.name

            # Load audio

            y, sr = librosa.load(
                temp_path,
                sr=22050
            )

            # Extract MFCC

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=40
            )

            # Create features

            features = np.mean(
                mfcc.T,
                axis=0
            )

            features = features.reshape(
                1,
                -1
            )

            # Predict

            prediction = model.predict(
                features
            )[0]

            emotion = emotion_labels.get(
                int(prediction),
                "Unknown"
            )

            # Confidence

            confidence = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    features
                )[0]

                confidence = max(
                    probabilities
                ) * 100

            # Display result

            st.success(
                f"🎭 Detected Emotion: {emotion}"
            )

            if confidence is not None:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            # Delete temporary file

            os.remove(temp_path)

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

else:

    st.info(
        "Please upload an audio file to begin."
    )

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "AI & Machine Learning Mini Project"
)