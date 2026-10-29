
"""Collect labeled hand-landmark samples for SignSpeak-AI."""

import csv
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

DATA_FILE = Path("data/raw/landmarks.csv")

# A-I plus K: ten static gesture classes.
LABELS = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
    "6": "F",
    "7": "G",
    "8": "H",
    "9": "I",
    "k": "K",
}

SAMPLES_EVERY_N_FRAMES = 5


def main():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    header = ["label"] + [
        f"{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

    file_exists = DATA_FILE.exists() and DATA_FILE.stat().st_size > 0

    if file_exists:
        existing_columns = list(pd.read_csv(DATA_FILE, nrows=0).columns)
        if existing_columns != header:
            print("Error: Existing CSV columns do not match the expected format.")
            print("No data was collected.")
            return

        totals = Counter(
            pd.read_csv(DATA_FILE, usecols=["label"])["label"].astype(str)
        )
    else:
        totals = Counter()

    session_counts = Counter()
    selected_label = None
    frame_count = 0

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    try:
        with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(header)

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
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(
                            frame, hand, mp_hands.HAND_CONNECTIONS
                        )

                        if selected_label is not None:
                            frame_count += 1

                            if frame_count % SAMPLES_EVERY_N_FRAMES == 0:
                                features = []
                                for point in hand.landmark:
                                    features.extend(
                                        [point.x, point.y, point.z]
                                    )

                                writer.writerow([selected_label] + features)
                                file.flush()
                                session_counts[selected_label] += 1
                                totals[selected_label] += 1

                    instructions = (
                        "1-9=A-I  K=K  P=Pause  Q=Quit"
                    )
                    cv2.putText(
                        frame, instructions, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2,
                    )

                    status = f"Selected: {selected_label or 'Paused'}"
                    cv2.putText(
                        frame, status, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2,
                    )

                    count_text = (
                        f"Session: {sum(session_counts.values())}  "
                        f"Total: {sum(totals.values())}"
                    )
                    cv2.putText(
                        frame, count_text, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (255, 255, 0), 2,
                    )

                    cv2.imshow("SignSpeak-AI | Data Collection", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break
                    elif key == ord("p"):
                        selected_label = None
                        frame_count = 0
                    elif chr(key) in LABELS:
                        selected_label = LABELS[chr(key)]
                        frame_count = 0

    except KeyboardInterrupt:
        print("\nCollection interrupted.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nDataset: {DATA_FILE.resolve()}")
    print(f"Samples collected this session: {dict(session_counts)}")
    print("\nCurrent dataset totals:")
    for label in sorted(totals):
        print(f"  {label}: {totals[label]}")


if __name__ == "__main__":
    main()
