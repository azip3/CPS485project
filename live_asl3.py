import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
import json

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'asl_model3_v2.keras'
CLASSES_FILE = 'class_names3.json'
IMAGE_SIZE   = (160, 160)
CONF_THRESH  = 0.3
TOP_N        = 3
# ─────────────────────────────────────────────────────────────────────────────

try:
    with open(CLASSES_FILE, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names: {class_names}")
except FileNotFoundError:
    class_names = [chr(65 + i) for i in range(26)] + ['space']
    print("Warning: class_names3.json not found — using fallback.")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

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
    top_guesses     = []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        # ── Square crop with generous padding ─────────────────────────────
        padding = 80
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)

        # Pad to square so resize doesn't distort the hand
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        if crop_w > crop_h:
            diff = crop_w - crop_h
            y_min = max(0, y_min - diff // 2)
            y_max = min(h, y_max + (diff - diff // 2))
        else:
            diff = crop_h - crop_w
            x_min = max(0, x_min - diff // 2)
            x_max = min(w, x_max + (diff - diff // 2))

        hand_crop = frame[y_min:y_max, x_min:x_max]
        if hand_crop.size == 0:
            continue

        hand_resized = cv2.resize(hand_crop, IMAGE_SIZE)
        hand_rgb     = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)

        # Feed raw [0,255] float32 — the model's internal preprocess_input
        # layer handles scaling to [-1,1] for MobileNetV2
        input_img = np.expand_dims(hand_rgb.astype(np.float32), axis=0)

        prediction      = model.predict(input_img, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence      = float(prediction[0][predicted_class])

        top_indices = np.argsort(prediction[0])[::-1][:TOP_N]
        top_guesses = [
            (class_names[i] if i < len(class_names) else "?", float(prediction[0][i]))
            for i in top_indices
        ]

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

    if top_guesses:
        cv2.putText(frame, "Top guesses:", (10, h - 20 - (TOP_N * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for idx, (lbl, conf) in enumerate(top_guesses):
            color = (0, 255, 0) if idx == 0 else (180, 180, 180)
            cv2.putText(frame, f"  {idx + 1}. {lbl}: {conf:.0%}",
                        (10, h - 20 - ((TOP_N - idx - 1) * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow('ASL Live Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()