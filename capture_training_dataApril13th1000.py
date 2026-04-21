"""
ASL Webcam Training Data Capture (224x224)
Captures hand crops for all ASL letters (A-Z + space)

Controls:
    SPACE      = Start / Pause capturing for current letter
    TAB        = Skip to next letter (saves whatever you have)
    A-Z keys   = Jump directly to that letter
    ESC        = Quit early (saves whatever you have)

Images APPEND to existing data — safe to run multiple times.
Saves to: webcam_training_data_224/<LETTER>/
Run from your project folder: python3 capture_training_data_224.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_SIZE          = (224, 224)
OUTPUT_DIR          = 'webcam_training_data_224_v21000images'
CAPTURES_PER_LETTER = 2500           # 1000 images per letter
CAPTURE_DELAY       = 0.15
TARGET_LETTERS      = [chr(65 + i) for i in range(26)]  # A-Z (all 26)
TARGET_LETTERS.append('space')                            # 27 total
# ─────────────────────────────────────────────────────────────────────────────

# Build lookup: key code -> index in TARGET_LETTERS
LETTER_TO_IDX = {}
for idx, letter in enumerate(TARGET_LETTERS):
    if len(letter) == 1:
        LETTER_TO_IDX[ord(letter.lower())] = idx
        LETTER_TO_IDX[ord(letter.upper())] = idx

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for letter in TARGET_LETTERS:
    os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)

def count_existing(letter):
    letter_dir = os.path.join(OUTPUT_DIR, letter)
    if os.path.exists(letter_dir):
        return len([f for f in os.listdir(letter_dir) if f.endswith('.jpg')])
    return 0

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
print("ASL WEBCAM TRAINING DATA CAPTURE (224x224)")
print("=" * 60)
print(f"Target letters: {', '.join(TARGET_LETTERS)}")
print(f"Images per letter: {CAPTURES_PER_LETTER}")
print(f"Save location: {OUTPUT_DIR}/")
print()
print("Controls:")
print("  SPACE      = Start / Pause capturing")
print("  TAB        = Skip to next letter")
print("  A-Z keys   = Jump directly to that letter")
print("  ESC        = Quit")
print()

# Show existing counts
print("Existing images:")
any_existing = False
for letter in TARGET_LETTERS:
    existing = count_existing(letter)
    if existing > 0:
        print(f"  {letter}: {existing} images")
        any_existing = True
if not any_existing:
    print("  (none)")
print("New images will APPEND (not overwrite).")
print("=" * 60)

def switch_to(idx):
    letter = TARGET_LETTERS[idx]
    existing = count_existing(letter)
    print(f"\nLetter: {letter} ({existing} existing)")
    return existing, 0

current_letter_idx = 0
capturing = False
existing_count, capture_count = switch_to(0)
last_capture_time = 0

while cap.isOpened() and current_letter_idx < len(TARGET_LETTERS):
    ret, frame = cap.read()
    if not ret:
        break

    current_letter = TARGET_LETTERS[current_letter_idx]
    total_for_letter = existing_count + capture_count

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_crop = None

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

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size > 0:
            hand_crop = cv2.resize(crop, IMAGE_SIZE)

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    remaining = CAPTURES_PER_LETTER - total_for_letter
    if capturing and hand_crop is not None and remaining > 0:
        now = time.time()
        if now - last_capture_time >= CAPTURE_DELAY:
            letter_dir = os.path.join(OUTPUT_DIR, current_letter)
            file_num = existing_count + capture_count
            filename = os.path.join(letter_dir, f"webcam_{current_letter}_{file_num:04d}.jpg")
            cv2.imwrite(filename, hand_crop)
            capture_count += 1
            total_for_letter = existing_count + capture_count
            last_capture_time = now

        if total_for_letter >= CAPTURES_PER_LETTER:
            capturing = False
            print(f"  Done! {current_letter}: {total_for_letter} total ({capture_count} new)")

    h, w, _ = frame.shape

    if capturing:
        color = (0, 0, 255)
        status = "CAPTURING... (SPACE to pause)"
    elif total_for_letter >= CAPTURES_PER_LETTER:
        color = (0, 200, 0)
        status = "COMPLETE! (TAB for next)"
    else:
        color = (255, 255, 255)
        status = "Ready (SPACE to start)"

    cv2.putText(frame, f"Show letter: {current_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, status, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    progress = min(1.0, total_for_letter / CAPTURES_PER_LETTER)
    bar_w = 300
    bar_x, bar_y, bar_h = 10, 100, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    fill_color = (0, 255, 0) if total_for_letter >= CAPTURES_PER_LETTER else (0, 165, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), fill_color, -1)
    cv2.putText(frame, f"{total_for_letter}/{CAPTURES_PER_LETTER} ({capture_count} new)",
                (bar_x + bar_w + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Time estimate
    if capturing and capture_count > 0:
        remaining_imgs = CAPTURES_PER_LETTER - total_for_letter
        est_seconds = int(remaining_imgs * CAPTURE_DELAY)
        est_min = est_seconds // 60
        est_sec = est_seconds % 60
        cv2.putText(frame, f"~{est_min}m {est_sec}s remaining",
                    (bar_x, bar_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, f"Letter {current_letter_idx + 1}/{len(TARGET_LETTERS)}: {current_letter}",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "A-Z=jump | TAB=next | ESC=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    if results.multi_hand_landmarks is None:
        cv2.putText(frame, "No hand detected", (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if hand_crop is not None:
        preview = cv2.resize(hand_crop, (120, 120))
        frame[10:130, w - 130:w - 10] = preview

    cv2.imshow('ASL Capture (224x224)', frame)

    raw_key = cv2.waitKey(1)
    key = raw_key & 0xFF

    if key == 27:
        print(f"\nQuitting. Saved {capture_count} new for '{current_letter}'.")
        break
    elif key == 32:
        if not capturing:
            capturing = True
            print(f"  Capturing '{current_letter}'...")
        else:
            capturing = False
            print(f"  Paused at {total_for_letter} total ({capture_count} new)")
    elif key == 9:
        capturing = False
        if capture_count > 0:
            print(f"  {current_letter}: saved {capture_count} new ({total_for_letter} total)")
        current_letter_idx += 1
        if current_letter_idx < len(TARGET_LETTERS):
            existing_count, capture_count = switch_to(current_letter_idx)
    elif key in LETTER_TO_IDX:
        jump_idx = LETTER_TO_IDX[key]
        if jump_idx != current_letter_idx:
            capturing = False
            if capture_count > 0:
                print(f"  {current_letter}: saved {capture_count} new ({total_for_letter} total)")
            current_letter_idx = jump_idx
            existing_count, capture_count = switch_to(current_letter_idx)

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("CAPTURE SUMMARY")
print("=" * 60)
total = 0
complete = 0
for letter in TARGET_LETTERS:
    count = count_existing(letter)
    status = "OK" if count >= CAPTURES_PER_LETTER else "NEED MORE"
    print(f"  {letter}: {count} images [{status}]")
    total += count
    if count >= CAPTURES_PER_LETTER:
        complete += 1
print(f"\nTotal: {total} images in '{OUTPUT_DIR}/'")
print(f"Complete: {complete}/{len(TARGET_LETTERS)} letters")
print(f"\nNext step: Zip and upload 'webcam_training_data_224' to Google Colab")