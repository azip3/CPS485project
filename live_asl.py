import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np

# Load your trained ASL model
model = tf.keras.models.load_model('asl_model.h5')

# MediaPipe Hands setup (classic API - works with 0.10.14)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,           # Video stream mode
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert BGR (OpenCV) to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    predicted_text = "No hand detected"

    if results.multi_hand_landmarks:
        # Get the first (and only) detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Calculate bounding box from landmarks
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))

        # Add padding so the crop includes the whole hand
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop the hand region
        hand_crop = frame[y_min:y_max, x_min:x_max]

        if hand_crop.size == 0:
            continue  # skip empty crops

        # Resize to 64×64 (your model's input size)
        hand_resized = cv2.resize(hand_crop, (64, 64))

        # Convert BGR → RGB and normalize to [0, 1]
        hand_rgb = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)
        hand_normalized = hand_rgb.astype(np.float32) / 255.0

        # Add batch dimension → shape (1, 64, 64, 3)
        input_img = np.expand_dims(hand_normalized, axis=0)

        # Predict
        prediction = model.predict(input_img, verbose=0)
        predicted_class = np.argmax(prediction[0])

        # Map class index to character
        if predicted_class < 10:
            char = str(predicted_class)                 # 0–9
        else:
            char = chr(97 + (predicted_class - 10))     # 10–35 → a–z

        predicted_text = f"Predicted: {char} (class {predicted_class})"

        # Draw green rectangle around the hand
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Show prediction text on top-left
    cv2.putText(frame, predicted_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ASL Live Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()