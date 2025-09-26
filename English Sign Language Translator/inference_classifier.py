import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
classes = model_dict['classes']

# Camera
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Sentence tracking
sentence = ""
predictions_queue = deque(maxlen=20)
last_added_char = ""
last_time_added = time.time()
ADD_LETTER_DELAY = 2.0  # seconds between adding letters

def draw_camera_box(img, x1, y1, x2, y2, color=(0, 0, 255), thickness=3):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

# Main loop
while True:
    data_aux = []
    x_, y_ = [], []
    ret, frame = cap.read()
    if not ret:
        continue
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1, y1 = int(min(x_) * W) - 20, int(min(y_) * H) - 20
            x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]
            predictions_queue.append(predicted_character)

            most_common_char, count = Counter(predictions_queue).most_common(1)[0]

            if count > 15 and (most_common_char != last_added_char or current_time - last_time_added > ADD_LETTER_DELAY):
                if most_common_char == 'Space':
                    sentence += ' '
                elif most_common_char == 'Backspace':
                    sentence = sentence[:-1]
                else:
                    sentence += most_common_char
                last_added_char = most_common_char
                last_time_added = current_time

            draw_camera_box(frame, x1, y1, x2, y2)

            cv2.putText(frame, most_common_char, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # Show sentence
    cv2.rectangle(frame, (20, 400), (620, 450), (255, 255, 255), -1)
    cv2.putText(frame, sentence, (30, 435),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    cv2.imshow('frame', frame)

    # Controls
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        last_added_char = ""
        predictions_queue.clear()

cap.release()
cv2.destroyAllWindows()
