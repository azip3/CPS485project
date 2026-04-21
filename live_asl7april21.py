import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import mediapipe as mp
import numpy as np
import json

# ── Config ────────────────────────────────────────────────────────────────────
# Model 1: 27-class general model (224x224)
WEIGHTS_MAIN    = 'asl_model4_27class.weights.h5'
#WEIGHTS_MAIN    = 'asl_model4_27class.weights (1).h5'

CLASSES_MAIN    = 'class_names4.json'
#CLASSES_MAIN    = 'class_names4 (1).json'

IMAGE_SIZE_MAIN = (224, 224)

# Model 2: 5-class specialist for confusable letters (160x160)
WEIGHTS_SPEC    = 'asl_model_5class.weights.h5'
CLASSES_SPEC    = 'class_names_5class.json'
IMAGE_SIZE_SPEC = (160, 160)

# When Model 1 predicts one of these AND confidence is below threshold,
# Model 2 gets consulted for a second opinion
CONFUSABLE      = ['A', 'M', 'N', 'O', 'T']
HANDOFF_THRESH  = 0.85

CONF_THRESH     = 0.3
TOP_N           = 3
AGREE_COUNT     = 6
# ─────────────────────────────────────────────────────────────────────────────

# ── Load class names ──────────────────────────────────────────────────────────
try:
    with open(CLASSES_MAIN, 'r') as f:
        class_names_main = json.load(f)
    print(f"Model 1: Loaded {len(class_names_main)} classes: {class_names_main}")
except FileNotFoundError:
    class_names_main = [chr(65 + i) for i in range(26)] + ['space']
    print("Warning: class_names4.json not found — using fallback.")

try:
    with open(CLASSES_SPEC, 'r') as f:
        class_names_spec = json.load(f)
    print(f"Model 2: Loaded {len(class_names_spec)} classes: {class_names_spec}")
except FileNotFoundError:
    class_names_spec = ['A', 'M', 'N', 'O', 'T']
    print("Warning: class_names_5class.json not found — using fallback.")

NUM_MAIN = len(class_names_main)
NUM_SPEC = len(class_names_spec)

# ── Build Model 1: 27-class (224x224) ────────────────────────────────────────
print("\nBuilding Model 1 (27-class, 224x224)...")
aug_main = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.3),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

base_main = applications.MobileNetV2(
    input_shape=IMAGE_SIZE_MAIN + (3,),
    include_top=False,
    weights='imagenet'
)
base_main.trainable = True
for layer in base_main.layers[:-20]:  # Matches training: top 20 layers
    layer.trainable = False

inp_main = keras.Input(shape=IMAGE_SIZE_MAIN + (3,))
x = aug_main(inp_main)
x = applications.mobilenet_v2.preprocess_input(x)
x = base_main(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out_main = layers.Dense(NUM_MAIN, activation='softmax')(x)

model_main = keras.Model(inp_main, out_main)
model_main.load_weights(WEIGHTS_MAIN)
print(f"Model 1 loaded: {WEIGHTS_MAIN}")

# ── Build Model 2: 5-class specialist (160x160) ──────────────────────────────
print("Building Model 2 (5-class specialist, 160x160)...")
aug_spec = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.3),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation_spec")

base_spec = applications.MobileNetV2(
    input_shape=IMAGE_SIZE_SPEC + (3,),
    include_top=False,
    weights='imagenet'
)
base_spec.trainable = True
for layer in base_spec.layers[:-30]:  # Specialist was trained with top 30
    layer.trainable = False

inp_spec = keras.Input(shape=IMAGE_SIZE_SPEC + (3,))
x2 = aug_spec(inp_spec)
x2 = applications.mobilenet_v2.preprocess_input(x2)
x2 = base_spec(x2, training=False)
x2 = layers.GlobalAveragePooling2D()(x2)
x2 = layers.Dense(512, activation='relu')(x2)
x2 = layers.Dropout(0.3)(x2)
x2 = layers.Dense(256, activation='relu')(x2)
x2 = layers.Dropout(0.3)(x2)
out_spec = layers.Dense(NUM_SPEC, activation='softmax')(x2)

model_spec = keras.Model(inp_spec, out_spec)
model_spec.load_weights(WEIGHTS_SPEC)
print(f"Model 2 loaded: {WEIGHTS_SPEC}")

print("\nTwo-stage system ready.")
print(f"Model 2 activates when Model 1 predicts {CONFUSABLE} with <{HANDOFF_THRESH:.0%} confidence.")

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

# ── Consecutive agreement state ───────────────────────────────────────────────
consecutive_count  = 0
last_raw_label     = ""
stable_label       = ""
stable_confidence  = 0.0
stable_top_guesses = []
stable_source      = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb_frame)

    predicted_text  = "No hand detected"
    confidence_text = ""
    source_text     = ""
    display_guesses = stable_top_guesses

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        padding = 80
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)

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

        # ── Model 1: 27-class prediction ──────────────────────────────────
        resized_main = cv2.resize(hand_crop, IMAGE_SIZE_MAIN)
        rgb_main     = cv2.cvtColor(resized_main, cv2.COLOR_BGR2RGB)
        input_main   = np.expand_dims(rgb_main.astype(np.float32), axis=0)

        pred_main       = model_main.predict(input_main, verbose=0)
        class_idx_main  = int(np.argmax(pred_main[0]))
        conf_main       = float(pred_main[0][class_idx_main])
        label_main      = class_names_main[class_idx_main]

        top_indices = np.argsort(pred_main[0])[::-1][:TOP_N]
        frame_guesses = [
            (class_names_main[i], float(pred_main[0][i]))
            for i in top_indices
        ]

        final_label = label_main
        final_conf  = conf_main
        source      = "Model 1"

        # ── Model 2: specialist override ──────────────────────────────────
        if label_main in CONFUSABLE and conf_main < HANDOFF_THRESH:
            resized_spec = cv2.resize(hand_crop, IMAGE_SIZE_SPEC)
            rgb_spec     = cv2.cvtColor(resized_spec, cv2.COLOR_BGR2RGB)
            input_spec   = np.expand_dims(rgb_spec.astype(np.float32), axis=0)

            pred_spec      = model_spec.predict(input_spec, verbose=0)
            class_idx_spec = int(np.argmax(pred_spec[0]))
            conf_spec      = float(pred_spec[0][class_idx_spec])
            label_spec     = class_names_spec[class_idx_spec]

            if conf_spec > conf_main:
                final_label = label_spec
                final_conf  = conf_spec
                source      = "Model 2"

                top_spec = np.argsort(pred_spec[0])[::-1][:TOP_N]
                frame_guesses = [
                    (class_names_spec[i], float(pred_spec[0][i]))
                    for i in top_spec
                ]

        # ── Consecutive agreement ─────────────────────────────────────────
        if final_label == last_raw_label:
            consecutive_count += 1
        else:
            consecutive_count = 1
            last_raw_label = final_label

        if consecutive_count >= AGREE_COUNT:
            stable_label       = final_label
            stable_confidence  = final_conf
            stable_top_guesses = frame_guesses
            stable_source      = source

        display_guesses = stable_top_guesses

        if stable_label:
            if stable_confidence >= CONF_THRESH:
                predicted_text  = f"Predicted: {stable_label}"
                confidence_text = f"Confidence: {stable_confidence:.0%}"
            else:
                predicted_text  = f"Low confidence: {stable_label}"
                confidence_text = f"({stable_confidence:.0%})"
            source_text = stable_source
        else:
            predicted_text = "Analyzing..."

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    h, w, _ = frame.shape

    cv2.putText(frame, predicted_text,  (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, confidence_text, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    if source_text:
        color = (255, 200, 0) if source_text == "Model 2" else (200, 200, 200)
        cv2.putText(frame, f"[{source_text}]", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    if display_guesses:
        cv2.putText(frame, "Top guesses:", (10, h - 20 - (TOP_N * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for idx, (lbl, conf) in enumerate(display_guesses):
            color = (0, 255, 0) if idx == 0 else (180, 180, 180)
            cv2.putText(frame, f"  {idx + 1}. {lbl}: {conf:.0%}",
                        (10, h - 20 - ((TOP_N - idx - 1) * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow('ASL Live Recognition v5 (Two-Stage)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()