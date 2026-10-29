
import json
import time
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf


# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "gesture_cnn.keras"
CLASSES_PATH = BASE_DIR / "models" / "gesture_cnn_classes.json"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


# --------------------------------------------------
# Load the trained CNN and class metadata
# --------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

CLASSES = metadata["classes"]
MEAN = np.asarray(metadata["mean"], dtype=np.float32)
STD = np.asarray(metadata["std"], dtype=np.float32)

# Avoid division by zero during normalization.
STD = np.where(STD < 1e-6, 1.0, STD)


# --------------------------------------------------
# MediaPipe Tasks API
# --------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Hand connections for drawing the detected landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --------------------------------------------------
# Text-to-speech
# --------------------------------------------------
def speak_text(text):
    """Speak the recognized text without blocking the webcam."""

    def worker():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            print(f"Text-to-speech error: {exc}")

    threading.Thread(target=worker, daemon=True).start()


# --------------------------------------------------
# Draw hand landmarks
# --------------------------------------------------
def draw_hand(frame, landmarks):
    """Draw detected hand landmarks and their connections."""

    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 100, 255), -1)


# --------------------------------------------------
# Main live recognition function
# --------------------------------------------------
def main():
    # Verify that all required model files exist.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found: {MODEL_PATH}"
        )

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class metadata not found: {CLASSES_PATH}"
        )

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand landmark model not found: {HAND_MODEL_PATH}"
        )

    # Configure MediaPipe hand tracking.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    text_buffer = ""
    current_prediction = ""
    last_spoken_text = ""

    confidence_threshold = 0.70

    print("\nSignSpeak-AI live recognition started.")
    print("Show a trained static hand gesture.")
    print("SPACE     : Add the current predicted letter")
    print("BACKSPACE : Delete the last letter")
    print("C         : Clear all text")
    print("T         : Speak the recognized text")
    print("Q         : Quit the application")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Warning: Could not read from webcam.")
                    break

                # Mirror the webcam image for a natural preview.
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                current_prediction = ""
                confidence = 0.0

                # ------------------------------------------
                # Detect the hand and predict its gesture.
                # ------------------------------------------
                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]

                    draw_hand(frame, hand_landmarks)

                    raw_features = np.array(
                        [
                            coordinate
                            for landmark in hand_landmarks
                            for coordinate in (
                                landmark.x,
                                landmark.y,
                                landmark.z,
                            )
                        ],
                        dtype=np.float32,
                    )

                    if raw_features.size == 63:
                        normalized = (raw_features - MEAN) / STD

                        cnn_input = normalized.reshape(
                            1, 63, 1
                        )

                        probabilities = model(
                            cnn_input,
                            training=False,
                        ).numpy()[0]

                        best_index = int(
                            np.argmax(probabilities)
                        )

                        confidence = float(
                            probabilities[best_index]
                        )

                        if confidence >= confidence_threshold:
                            current_prediction = CLASSES[best_index]

                    # Display the prediction only.
                    # Do NOT append letters automatically.
                    if current_prediction:
                        cv2.putText(
                            frame,
                            f"Prediction: {current_prediction} "
                            f"({confidence:.0%})",
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            frame,
                            "Prediction: Not confident",
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                else:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # ------------------------------------------
                # Display the manually composed text.
                # ------------------------------------------
                cv2.putText(
                    frame,
                    f"Text: {text_buffer[-35:]}",
                    (20, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "SPACE: add | BACKSPACE: delete | C: clear",
                    (20, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(
                    "SignSpeak-AI - Live Recognition",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                # ------------------------------------------
                # Keyboard controls
                # ------------------------------------------
                if key == ord("q"):
                    break

                elif key == ord("c"):
                    text_buffer = ""
                    last_spoken_text = ""

                elif key == 8:  # Backspace
                    text_buffer = text_buffer[:-1]
                    last_spoken_text = ""

                elif key == 32 and current_prediction:
                    # Add a letter ONLY when SPACE is pressed.
                    text_buffer += current_prediction
                    last_spoken_text = ""

                elif key == ord("t") and text_buffer.strip():
                    if text_buffer != last_spoken_text:
                        speak_text(text_buffer)
                        last_spoken_text = text_buffer

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
