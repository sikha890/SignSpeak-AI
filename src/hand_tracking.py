
"""Real-time hand tracking for SignSpeak-AI."""

import cv2
import mediapipe as mp


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Error: Could not read webcam frame.")
                    break

                # Mirror the webcam image.
                frame = cv2.flip(frame, 1)

                # OpenCV uses BGR; MediaPipe expects RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                    message = "Hand detected"
                    color = (0, 255, 0)
                else:
                    message = "No hand detected"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    message,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

                cv2.imshow("SignSpeak-AI | Hand Tracking", frame)

                # Press q to quit.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
