import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
prev_level = -1  # Track volume level to avoid repeated presses

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        # Thumb tip
        x1 = int(hand.landmark[4].x * w)
        y1 = int(hand.landmark[4].y * h)

        # Index tip
        x2 = int(hand.landmark[8].x * w)
        y2 = int(hand.landmark[8].y * h)

        # Distance between fingers
        dist = math.hypot(x2 - x1, y2 - y1)

        # Map distance 20–200 → 0–10 volume levels
        level = int(np.interp(dist, [20, 200], [0, 10]))

        # Draw UI
        cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, f"Level: {level}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Control volume only when level changes
        if level != prev_level:
            if level > prev_level:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

            prev_level = level

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, "Gesture Volume Control", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
