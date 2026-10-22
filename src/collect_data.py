
"""Collect labeled hand-landmark samples for SignSpeak-AI."""

import csv
from pathlib import Path

import cv2
import mediapipe as mp

# Run this script from the project root.
DATA_FILE = Path("data/raw/landmarks.csv")
LABELS = {"1": "A", "2": "B", "3": "C"}


def main():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = DATA_FILE.exists() and DATA_FILE.stat().st_size > 0

    header = ["label"] + [
        f"{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    counts = {"A": 0, "B": 0, "C": 0}
    selected_label = None
    frame_count = 0

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

                            # Save one sample every five frames.
                            if frame_count % 5 == 0:
                                features = []
                                for point in hand.landmark:
                                    features.extend(
                                        [point.x, point.y, point.z]
                                    )

                                writer.writerow([selected_label] + features)
                                file.flush()
                                counts[selected_label] += 1

                    instructions = "1=A  2=B  3=C  0=Pause  Q=Quit"
                    cv2.putText(
                        frame, instructions, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0), 2
                    )

                    status = f"Selected: {selected_label or 'Paused'}"
                    cv2.putText(
                        frame, status, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2
                    )

                    count_text = (
                        f"A: {counts['A']}  B: {counts['B']}  "
                        f"C: {counts['C']}"
                    )
                    cv2.putText(
                        frame, count_text, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 0), 2
                    )

                    cv2.imshow("SignSpeak-AI | Data Collection", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break
                    elif key in map(ord, LABELS):
                        selected_label = LABELS[chr(key)]
                        frame_count = 0
                    elif key == ord("0"):
                        selected_label = None

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Dataset file: {DATA_FILE.resolve()}")
    print(f"Samples collected this session: {counts}")


if __name__ == "__main__":
    main()
