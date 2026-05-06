# ══════════════════════════════════════════════════════════════════════════════
# ASL Live Recognition — Final (Single model, 256×256)
# ══════════════════════════════════════════════════════════════════════════════
#
# Changes from live_asl5.py:
#   * Two-stage specialist system removed entirely. One model, one pass.
#   * Matches the 256×256 training architecture (build_model identical to
#     build_model() in train_asl_Final.py — see Bug 8).
#   * No RandomFlip in the augmentation stack (matches training graph).
#   * Uses model(x, training=False) instead of model.predict() — significantly
#     faster per frame on M1.
#   * Fixed square-crop bug: when the hand is near a frame edge, we now
#     zero-pad the crop to square AFTER cropping, rather than letting the
#     bbox clip silently and cv2.resize stretch a non-square rectangle.
#   * Top-N display updates in the same frame as the agreement flip (old
#     code had a one-frame lag).
#
# Requirements unchanged: Python 3.9 on M1, tensorflow-macos +
# tensorflow-metal for GPU acceleration (strongly recommended).

import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import mediapipe as mp

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS_FILE         = 'asl_model_Final.weights.h5'
CLASSES_FILE         = 'class_names_Final.json'
IMAGE_SIZE           = (256, 256)
FINETUNE_LAYERS      = 40    # MUST match the training script exactly
DROPOUT_RATE         = 0.3   # MUST match the training script exactly

# Display & smoothing
CONF_THRESH  = 0.30   # below this, show "Low confidence: X" instead of "Predicted: X"
TOP_N        = 3
AGREE_COUNT  = 6      # frames of identical top-1 before the display updates
PADDING      = 80     # pixels of padding around the MediaPipe bbox

# ══════════════════════════════════════════════════════════════════════════════
# Architecture builder — MUST match train_asl_Final.py exactly (Bug 8)
# ══════════════════════════════════════════════════════════════════════════════

def build_augmentation():
    """Matches train_asl_Final.py exactly. RandomFlip intentionally absent —
    ASL signs are hand-specific. Augmentation is a no-op at inference; it's
    included purely so the graph structure matches training for weight
    loading."""
    return keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomBrightness(0.4),
        layers.RandomContrast(0.4),
    ], name="augmentation")


def build_model(num_classes):
    """Identical to build_model() in train_asl_Final.py."""
    aug = build_augmentation()
    base = applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = True
    for layer in base.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = aug(inputs)
    x = applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    # dtype='float32' was used in training for mixed precision; fine to omit
    # here since inference runs in float32 anyway.
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

# ══════════════════════════════════════════════════════════════════════════════
# Load class names
# ══════════════════════════════════════════════════════════════════════════════

try:
    with open(CLASSES_FILE, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} classes: {class_names}")
except FileNotFoundError:
    class_names = [chr(65 + i) for i in range(26)] + ['space']
    print(f"Warning: {CLASSES_FILE} not found — using fallback order.")

NUM_CLASSES = len(class_names)

# ══════════════════════════════════════════════════════════════════════════════
# Build & load weights
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nBuilding model ({NUM_CLASSES}-class, {IMAGE_SIZE})...")
model = build_model(NUM_CLASSES)
model.load_weights(WEIGHTS_FILE)
print(f"Model loaded: {WEIGHTS_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def square_crop_with_padding(frame, x_min, y_min, x_max, y_max, target_size):
    """Crop the hand region and produce a truly square output of target_size.

    Guarantees squareness by zero-padding when the desired square bbox
    exceeds frame bounds. The previous implementation clipped the bbox at the
    edges and let cv2.resize silently stretch a non-square rectangle into a
    square, distorting hand proportions whenever the hand was near the edge
    of the frame.

    Must remain identical to the version in capture_training_data_Final.py
    so train-time and inference-time crops come from the same distribution.
    """
    h, w = frame.shape[:2]

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    side   = max(bbox_w, bbox_h)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    sq_x_min = cx - side // 2
    sq_y_min = cy - side // 2
    sq_x_max = sq_x_min + side
    sq_y_max = sq_y_min + side

    pad_left   = max(0, -sq_x_min)
    pad_top    = max(0, -sq_y_min)
    pad_right  = max(0, sq_x_max - w)
    pad_bottom = max(0, sq_y_max - h)

    x0 = max(0, sq_x_min)
    y0 = max(0, sq_y_min)
    x1 = min(w, sq_x_max)
    y1 = min(h, sq_y_max)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    return cv2.resize(crop, target_size)


def predict_fast(bgr_crop):
    """Run a single frame through the model. Uses model(x) not model.predict()
    — much faster for single-frame real-time inference on M1."""
    rgb   = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    x     = np.expand_dims(rgb.astype(np.float32), axis=0)
    probs = model(x, training=False).numpy()[0]
    return probs

# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe setup
# ══════════════════════════════════════════════════════════════════════════════

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open webcam.")
    raise SystemExit(1)

print("\nWebcam opened. Press 'q' to quit.")

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

consecutive_count  = 0
last_raw_label     = ""
stable_label       = ""
stable_confidence  = 0.0
stable_top_guesses = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb_frame)

    predicted_text  = "No hand detected"
    confidence_text = ""
    display_guesses = stable_top_guesses

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _  = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min = int(min(x_coords)) - PADDING
        y_min = int(min(y_coords)) - PADDING
        x_max = int(max(x_coords)) + PADDING
        y_max = int(max(y_coords)) + PADDING

        crop = square_crop_with_padding(
            frame, x_min, y_min, x_max, y_max, IMAGE_SIZE,
        )
        if crop is None:
            continue

        # Forward pass
        probs     = predict_fast(crop)
        class_idx = int(np.argmax(probs))
        conf      = float(probs[class_idx])
        label     = class_names[class_idx]

        top_indices   = np.argsort(probs)[::-1][:TOP_N]
        frame_guesses = [(class_names[i], float(probs[i]))
                         for i in top_indices]

        # Agreement smoothing
        if label == last_raw_label:
            consecutive_count += 1
        else:
            consecutive_count = 1
            last_raw_label    = label

        if consecutive_count >= AGREE_COUNT:
            stable_label       = label
            stable_confidence  = conf
            stable_top_guesses = frame_guesses

        # Show top guesses from the CURRENT frame, not stale ones
        display_guesses = frame_guesses

        if stable_label:
            if stable_confidence >= CONF_THRESH:
                predicted_text  = f"Predicted: {stable_label}"
                confidence_text = f"Confidence: {stable_confidence:.0%}"
            else:
                predicted_text  = f"Low confidence: {stable_label}"
                confidence_text = f"({stable_confidence:.0%})"
        else:
            predicted_text = "Analyzing..."

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        disp_x0 = max(0, x_min)
        disp_y0 = max(0, y_min)
        disp_x1 = min(w, x_max)
        disp_y1 = min(h, y_max)
        cv2.rectangle(frame, (disp_x0, disp_y0), (disp_x1, disp_y1), (0, 255, 0), 2)

    h, w, _ = frame.shape
    cv2.putText(frame, predicted_text,  (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, confidence_text, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    if display_guesses:
        cv2.putText(frame, "Top guesses:", (10, h - 20 - (TOP_N * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for idx, (lbl, c) in enumerate(display_guesses):
            color = (0, 255, 0) if idx == 0 else (180, 180, 180)
            cv2.putText(frame, f"  {idx + 1}. {lbl}: {c:.0%}",
                        (10, h - 20 - ((TOP_N - idx - 1) * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow('ASL Live Recognition (Final, 256x256)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()