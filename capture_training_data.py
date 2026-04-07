"""
ASL Webcam Training Data Capture
Captures hand crops for problem letters: A, N, M, O, T

Controls:
    SPACE  = Start / Pause capturing for current letter
    N      = Skip to next letter (saves whatever you have)
    Q      = Quit early (saves whatever you have)

Images are saved to: webcam_training_data/<LETTER>/
Run from your project folder: python3 capture_training_data.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_SIZE          = (160, 160)     # Match model 3 training resolution
OUTPUT_DIR          = 'webcam_training_data'
CAPTURES_PER_LETTER = 300            # Target images per letter
CAPTURE_DELAY       = 0.15           # Seconds between captures
TARGET_LETTERS      = ['A', 'M', 'N', 'O', 'T']
# ─────────────────────────────────────────────────────────────────────────────

# Create output directory automatically
os.makedirs(OUTPUT_DIR, exist_ok=True)
for letter in TARGET_LETTERS:
    os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)

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

print("=" * 60)
print("ASL WEBCAM TRAINING DATA CAPTURE")
print("=" * 60)
print(f"Target letters: {', '.join(TARGET_LETTERS)}")
print(f"Images per letter: {CAPTURES_PER_LETTER}")
print(f"Save location: {OUTPUT_DIR}/")
print()
print("Controls:")
print("  SPACE  = Start / Pause capturing")
print("  N      = Skip to next letter")
print("  Q      = Quit")
print("=" * 60)
print()
print(f"First letter: {TARGET_LETTERS[0]}")
print("Hold up the sign and press SPACE to start capturing.")

current_letter_idx = 0
capturing = False
capture_count = 0
last_capture_time = 0

while cap.isOpened() and current_letter_idx < len(TARGET_LETTERS):
    ret, frame = cap.read()
    if not ret:
        break

    current_letter = TARGET_LETTERS[current_letter_idx]

    # Process hand detection on clean frame BEFORE drawing landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_crop = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        # Square crop with generous padding (matches live_asl3.py)
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

        # Crop BEFORE drawing landmarks so saved images are clean
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size > 0:
            hand_crop = cv2.resize(crop, IMAGE_SIZE)

        # Draw landmarks on display frame AFTER cropping
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Save crops while capturing
    if capturing and hand_crop is not None and capture_count < CAPTURES_PER_LETTER:
        now = time.time()
        if now - last_capture_time >= CAPTURE_DELAY:
            letter_dir = os.path.join(OUTPUT_DIR, current_letter)
            filename = os.path.join(letter_dir, f"webcam_{current_letter}_{capture_count:04d}.jpg")
            cv2.imwrite(filename, hand_crop)
            capture_count += 1
            last_capture_time = now

        # Auto-stop when target reached
        if capture_count >= CAPTURES_PER_LETTER:
            capturing = False
            print(f"  Done! Captured {capture_count}/{CAPTURES_PER_LETTER} for '{current_letter}'")
            print(f"  Press N to move to next letter, or SPACE to capture more.")

    # ── Display info ──────────────────────────────────────────────────────
    h, w, _ = frame.shape

    # Current letter and status
    if capturing:
        color = (0, 0, 255)
        status = "CAPTURING... (SPACE to pause)"
    elif capture_count >= CAPTURES_PER_LETTER:
        color = (0, 200, 0)
        status = "COMPLETE! (N for next letter)"
    else:
        color = (255, 255, 255)
        status = "Ready (SPACE to start)"

    cv2.putText(frame, f"Show letter: {current_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, status, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Progress bar
    progress = capture_count / CAPTURES_PER_LETTER
    bar_w = 300
    bar_h = 20
    bar_x = 10
    bar_y = 100
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    fill_color = (0, 255, 0) if capture_count >= CAPTURES_PER_LETTER else (0, 165, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), fill_color, -1)
    cv2.putText(frame, f"{capture_count}/{CAPTURES_PER_LETTER}", (bar_x + bar_w + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Overall progress
    overall = f"Letter {current_letter_idx + 1}/{len(TARGET_LETTERS)}: {current_letter}"
    cv2.putText(frame, overall, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # No hand warning
    if results.multi_hand_landmarks is None:
        cv2.putText(frame, "No hand detected", (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show preview of crop in corner
    if hand_crop is not None:
        preview = cv2.resize(hand_crop, (120, 120))
        frame[10:130, w - 130:w - 10] = preview

    cv2.imshow('ASL Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"\nQuitting early. Saved {capture_count} images for '{current_letter}'.")
        break
    elif key == ord(' '):
        if not capturing:
            capturing = True
            print(f"  Capturing '{current_letter}'...")
        else:
            capturing = False
            print(f"  Paused at {capture_count}/{CAPTURES_PER_LETTER}")
    elif key == ord('n'):
        capturing = False
        print(f"  Moving on from '{current_letter}' ({capture_count} images saved)")
        current_letter_idx += 1
        capture_count = 0
        if current_letter_idx < len(TARGET_LETTERS):
            print(f"\nNext letter: {TARGET_LETTERS[current_letter_idx]}")
            print("Hold up the sign and press SPACE to start capturing.")

cap.release()
cv2.destroyAllWindows()

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CAPTURE SUMMARY")
print("=" * 60)
total = 0
for letter in TARGET_LETTERS:
    letter_dir = os.path.join(OUTPUT_DIR, letter)
    if os.path.exists(letter_dir):
        count = len([f for f in os.listdir(letter_dir) if f.endswith('.jpg')])
        status = "OK" if count >= CAPTURES_PER_LETTER else "LOW"
        print(f"  {letter}: {count} images [{status}]")
        total += count
    else:
        print(f"  {letter}: 0 images [MISSING]")
print(f"\nTotal: {total} images saved to '{OUTPUT_DIR}/'")
print("\nNext step: Upload the 'webcam_training_data' folder to Google Colab")
print("and run the training script.")
