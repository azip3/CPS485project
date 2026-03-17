import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
import json

MODEL_PATH  = 'asl_model2_v2.keras'
IMAGE_SIZE  = (128, 128)
CONF_THRESH = 0.6

try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names.")
except FileNotFoundError:
    class_names = [str(i) for i in range(10)] + [chr(97 + i) for i in range(26)]
    print("Warning: class_names.json not found — using fallback.")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded.")

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

SMOOTHING_WINDOW = 5
pred_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb_frame)

    predicted_text  = "No hand detected"
    confidence_text = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        padding = 40
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)

        hand_crop = frame[y_min:y_max, x_min:x_max]
        if hand_crop.size == 0:
            continue

        hand_resized = cv2.resize(hand_crop, IMAGE_SIZE)
        hand_rgb     = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)

        input_img = tf.keras.applications.mobilenet_v2.preprocess_input(
            hand_rgb.astype(np.float32)
        )
        input_img = np.expand_dims(input_img, axis=0)

        prediction      = model.predict(input_img, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence      = float(prediction[0][predicted_class])

        # ── DEBUG ──────────────────────────────────────────────────────────
        print(f"Input range: min={input_img.min():.3f}, max={input_img.max():.3f}")
        print(f"Top 5 predictions:")
        top5 = np.argsort(prediction[0])[::-1][:5]
        for i in top5:
            lbl = class_names[i] if i < len(class_names) else "?"
            print(f"  class {i} ({lbl}): {prediction[0][i]:.4f}")
        print("──────────────────────────────────────────────────────────────")
        # ───────────────────────────────────────────────────────────────────

        pred_buffer.append(predicted_class)
        if len(pred_buffer) > SMOOTHING_WINDOW:
            pred_buffer.pop(0)
        smoothed_class = max(set(pred_buffer), key=pred_buffer.count)

        label = class_names[smoothed_class] if smoothed_class < len(class_names) else "?"

        if confidence >= CONF_THRESH:
            predicted_text  = f"Predicted: {label}"
            confidence_text = f"Confidence: {confidence:.0%}"
        else:
            predicted_text  = f"Low confidence: {label}"
            confidence_text = f"({confidence:.0%})"

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.putText(frame, predicted_text,  (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, confidence_text, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    cv2.imshow('ASL Live Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()