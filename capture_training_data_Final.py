# ══════════════════════════════════════════════════════════════════════════════
# ASL Webcam Capture — Final
# ══════════════════════════════════════════════════════════════════════════════
#
# Captures clean 256×256 hand crops for training and validation. Uses the
# IDENTICAL crop pipeline as live_asl_Final.py, so training-time crops match
# inference-time crops exactly.
#
# Changes from capture_training_data_224.py:
#   * Saves at 256×256 (matches train_asl_Final.py).
#   * Uses square_crop_with_padding (zero-pads when hand near frame edge),
#     matching the live script. The old logic clipped the bbox at the edges
#     and let cv2.resize stretch a non-square rectangle into a square, which
#     means hands near the edge of the frame ended up subtly distorted in
#     training data — a quiet version of the same distortion bug you'd get
#     at inference time.
#   * Identical 80px PADDING to live_asl_Final.py.
#
# Controls (focus the OpenCV window first — click on it):
#   SPACE  — start / pause capturing for the current letter
#   TAB    — advance to the next letter
#   A–Z    — jump directly to that letter
#   ESC    — quit
#
# Append mode is on by default — safe to run multiple times. Existing files
# are left in place; new captures get unique timestamped names.
#
# For session-separated validation:
#   1. First run: OUTPUT_DIR = 'webcam_training_data_Final'  → main training set
#   2. Different day, different lighting, different room:
#      OUTPUT_DIR = 'webcam_val_Final'                       → held-out val
#   3. (Optional) Yet another session:
#      OUTPUT_DIR = 'webcam_test_Final'                      → final test set
#                                                              (don't peek until demo)

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_SIZE          = (256, 256)                          # MUST match training
OUTPUT_DIR          = 'webcam_training_data_Final'        # change for val/test sessions
CAPTURES_PER_LETTER = 2000
CAPTURE_DELAY       = 0.15                                # seconds between captures
PADDING             = 80                                  # MUST match live_asl_Final.py

TARGET_LETTERS = [chr(65 + i) for i in range(26)]
TARGET_LETTERS.append('space')

# Map ASCII keycodes (lower & upper) to letter indices, plus space (' ' → 'space')
LETTER_TO_IDX = {}
for idx, letter in enumerate(TARGET_LETTERS):
    if len(letter) == 1:
        LETTER_TO_IDX[ord(letter.lower())] = idx
        LETTER_TO_IDX[ord(letter.upper())] = idx
LETTER_TO_IDX[ord(' ')] = TARGET_LETTERS.index('space')   # space bar → 'space' label

# ══════════════════════════════════════════════════════════════════════════════
# Crop helper — IDENTICAL to live_asl_Final.py.square_crop_with_padding
# ══════════════════════════════════════════════════════════════════════════════

def square_crop_with_padding(frame, x_min, y_min, x_max, y_max, target_size):
    """Crop the hand region and produce a truly square output of target_size.

    Guarantees squareness by zero-padding when the desired square bbox
    exceeds frame bounds. Must remain identical to the version in
    live_asl_Final.py — otherwise capture-time and inference-time crops
    diverge and the model sees a distribution shift.
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

# ══════════════════════════════════════════════════════════════════════════════
# Filesystem setup
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)
for letter in TARGET_LETTERS:
    os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)


def count_existing(letter):
    letter_dir = os.path.join(OUTPUT_DIR, letter)
    if os.path.exists(letter_dir):
        return len([f for f in os.listdir(letter_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return 0

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

print(f"Output: {OUTPUT_DIR}/  (target: {CAPTURES_PER_LETTER} per letter)")
print("\nControls (click the video window first):")
print("  SPACE   start/pause capturing")
print("  TAB     next letter")
print("  A–Z     jump to letter")
print("  ESC     quit\n")

# ══════════════════════════════════════════════════════════════════════════════
# Capture state
# ══════════════════════════════════════════════════════════════════════════════

current_idx       = 0
capturing         = False
last_capture_time = 0.0
session_count     = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb_frame)

    current_letter = TARGET_LETTERS[current_idx]
    existing       = count_existing(current_letter)
    remaining      = max(0, CAPTURES_PER_LETTER - existing)

    hand_seen = False
    crop_for_save = None
    bbox_for_display = None

    if results.multi_hand_landmarks:
        hand_seen = True
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _  = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min = int(min(x_coords)) - PADDING
        y_min = int(min(y_coords)) - PADDING
        x_max = int(max(x_coords)) + PADDING
        y_max = int(max(y_coords)) + PADDING

        # IMPORTANT: build the crop BEFORE drawing landmarks on the frame, so
        # the saved image is clean (no green skeleton overlay).
        crop_for_save    = square_crop_with_padding(
            frame, x_min, y_min, x_max, y_max, IMAGE_SIZE,
        )
        bbox_for_display = (max(0, x_min), max(0, y_min),
                            min(w, x_max),  min(h, y_max))

    # Save (only if capturing, hand seen, crop succeeded, delay elapsed,
    # and we haven't hit the per-letter cap)
    now = time.time()
    if (capturing
            and hand_seen
            and crop_for_save is not None
            and remaining > 0
            and (now - last_capture_time) >= CAPTURE_DELAY):
        ts        = int(now * 1000)
        filename  = f"{current_letter}_{ts}.jpg"
        save_path = os.path.join(OUTPUT_DIR, current_letter, filename)
        cv2.imwrite(save_path, crop_for_save)
        last_capture_time = now
        session_count    += 1
        existing         += 1
        remaining         = max(0, CAPTURES_PER_LETTER - existing)

    # ── HUD overlay (drawn AFTER any save, so the saved file stays clean) ──
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0],
                                  mp_hands.HAND_CONNECTIONS)
    if bbox_for_display is not None:
        cv2.rectangle(frame,
                      (bbox_for_display[0], bbox_for_display[1]),
                      (bbox_for_display[2], bbox_for_display[3]),
                      (0, 255, 0), 2)

    status_color = (0, 255, 0) if capturing else (0, 165, 255)
    status_text  = "CAPTURING" if capturing else "PAUSED"
    cv2.putText(frame, f"Letter: {current_letter}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"{status_text}  ({existing}/{CAPTURES_PER_LETTER})", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Session this run: {session_count}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if not hand_seen:
        cv2.putText(frame, "No hand detected", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('ASL Capture (Final)', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:                                    # ESC → quit
        break
    elif key == ord(' '):                            # SPACE → toggle capture
        # Special case: only treat SPACE as "save into 'space' folder" when
        # 'space' is already the active letter; otherwise SPACE means pause/resume.
        if current_letter == 'space':
            capturing = not capturing
        else:
            capturing = not capturing
    elif key == 9:                                   # TAB → next letter
        current_idx = (current_idx + 1) % len(TARGET_LETTERS)
        capturing = False
    elif key in LETTER_TO_IDX:                       # A-Z → jump
        # Don't let SPACE-as-keypress also jump to 'space' — already handled above
        if key != ord(' '):
            current_idx = LETTER_TO_IDX[key]
            capturing = False

cap.release()
cv2.destroyAllWindows()

# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSession complete. {session_count} new images saved to {OUTPUT_DIR}/")
print("\nPer-letter totals:")
for letter in TARGET_LETTERS:
    count = count_existing(letter)
    bar = "#" * min(40, count // 25)
    print(f"  {letter:>5}: {count:>4}  {bar}")
