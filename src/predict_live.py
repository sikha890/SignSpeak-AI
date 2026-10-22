
"""Recognize A, B, and C hand gestures using a webcam."""

from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "gesture_model.joblib"

CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 7


def extract_landmarks(hand_landmarks):
    """Convert 21 hand landmarks into 63 model features."""
    features = []

    for point in hand_landmarks.landmark:
        features.extend([point.x, point.y, point.z])

    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Run python src/train_model.py first."
        )

    model = joblib.load(MODEL_FILE)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Error: Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                results = hands.process(rgb_frame)

                display_label = "No hand detected"
                display_color = (0, 0, 255)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    features = extract_landmarks(hand)
                    probabilities = model.predict_proba(features)[0]
                    best_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[best_index])
                    predicted_label = str(model.classes_[best_index])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        prediction_history.append(predicted_label)

                        stable_label = Counter(
                            prediction_history
                        ).most_common(1)[0][0]

                        display_label = (
                            f"Gesture: {stable_label} "
                            f"({confidence:.1%})"
                        )
                        display_color = (0, 255, 0)
                    else:
                        prediction_history.clear()
                        display_label = "Gesture: Unknown"
                        display_color = (0, 165, 255)
                else:
                    prediction_history.clear()

                cv2.putText(
                    frame,
                    display_label,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    display_color,
                    2,
                )

                cv2.putText(
                    frame,
                    "Recognizes A/B/C | Press Q to quit",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow(
                    "SignSpeak-AI | Live Gesture Recognition",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
