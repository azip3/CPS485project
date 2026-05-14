import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import mediapipe as mp
import numpy as np
import json
 
WEIGHTS_PATH = 'asl_modelDemonstration_v2.weights.h5'
IMAGE_SIZE   = (224, 224)
CONF_THRESH  = 0.6
NUM_CLASSES  = 36  # 0-9 + a-z
TOP_N        = 3
 
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names.")
except FileNotFoundError:
    class_names = [str(i) for i in range(10)] + [chr(97 + i) for i in range(26)]
    print("Warning: class_names.json not found — using fallback.")
 
# ── Rebuild the exact same architecture as training ───────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
 
base_model = applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True  # Must match the state after fine-tuning
fine_tune_at = len(base_model.layers) // 2
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
 
inputs = layers.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)
x = applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.6)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
 
model = keras.Model(inputs, outputs)
model.load_weights(WEIGHTS_PATH)
print(f"Model weights loaded from {WEIGHTS_PATH}")
 
# ── MediaPipe setup ───────────────────────────────────────────────────────────
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
 
SMOOTHING_WINDOW = 15
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
 
    h, w, _ = frame.shape
 
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