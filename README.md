# CPS485project

## Description
(Basic Proposal/subject to change)
My project idea is a live ASL gesture recognition application. It would use a Convolutional Neural Network  to classify American Sign Language hand gestures from webcam video input. The system would detect hand landmarks, processes frames, and displays the predicted sign in real time.

Key features:
- Real-time gesture detection via webcam
- Trained neural network for high-accuracy classification
- User-friendly interface for live demonstration
- Possible visualization of training progress (loss/accuracy curves)

possible timeline until midterm
(weeks starting from February 3)
- Week 1-2/Data Acquisition and Model Development: Collect ASL datasets from Kaggle and start buliding CNN architecture;Train on static images; Initial hyperparameter tuning.
- Week 3-4/Integration & Real-time Testing: Link the model to OpenCV; Implement MediaPipe for landmark detection.
- Week 4-5/UI Design & Optimization: Build the user interface; Optimize for latency; Finalize loss/accuracy visualizations.
- Week 6( just week 6 if midterm is week 7): Final bug fixes

## 2/3 - 2/10

### 1. Building the CNN in Colab and initial training

- Set up development environment in Google Colab (Python 3, TensorFlow/Keras)
- Loaded ASL image dataset from directory (`asl_dataset`) using `image_dataset_from_directory`
  - Image size: 64×64 RGB
  - Batch size: 32
  - 20% validation split, categorical labels
  - Total classes: 36 (0-9 digits + a-z letters)
- Added basic data augmentation: RandomFlip (horizontal), RandomRotation (0.1), RandomZoom (0.1)
- Built a custom CNN architecture:
  - Data augmentation layer
  - Rescaling (1/255)
  - 3 convolutional blocks: Conv2D (32 → 64 → 128 filters, 3×3 kernel, ReLU) + MaxPooling2D (2×2) after each
  - Flatten → Dense 128 (ReLU) → Dropout 0.6 → Dense 37 (softmax)
  - Compiled with RMSprop optimizer, categorical_crossentropy loss, accuracy metric
- Trained for 6 epochs on training dataset with validation
- Final validation accuracy: ~50% (now 75% because fixed bad training data, although this model is not the one in repo) (exact value from `model.evaluate`: see printed Test Accuracy)
- Saved trained model as `asl_model.h5` for local use
- Challenges: Accuracy still moderate due to limited epochs, small dataset, and no transfer learning yet
- Next steps: Try more epochs, hyperparameter tuning, or switch to transfer learning (e.g. MobileNetV2 backbone) for better performance

### 2. Basic skeleton script (.py file)

- Created `live_asl.py` as the walking skeleton / MVP
- Loaded saved model (`asl_model.h5`) using `tf.keras.models.load_model`
- Used OpenCV (`cv2`) to capture webcam frames in real time
- Integrated MediaPipe Hands (v0.10.14) for hand landmark detection:
  - Configured for video stream (static_image_mode=False)
  - Limited to 1 hand, min detection/tracking confidence 0.5
- Implemented end-to-end pipeline:
  - Read frame → convert BGR to RGB
  - Detect hand landmarks → compute bounding box from landmarks
  - Crop hand region with padding → resize to 64×64 → normalize (0-1)
  - Add batch dimension → run model prediction
  - Map predicted class (0-36) to character (0-9 → '0'-'9', 10-35 → 'a'-'z')
  - Overlay prediction text and green bounding box on frame
  - Display with `cv2.imshow` — press 'q' to quit
- Current status: Camera opens successfully (light on), model loads, pipeline logic complete
- This establishes the core end-to-end connection: webcam input → hand detection → model inference → visual output on screen

### 3. Paper read: AlexNet

- Read: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012)  
  Link: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
- Why this paper: Foundational work that popularized deep CNNs for image classification and provides strong justification for using a CNN in this project
- Key takeaways:
  - Demonstrated that deep convolutional networks with ReLU activations, dropout, and data augmentation dramatically outperform previous methods on large-scale image recognition (ImageNet)
  - Introduced techniques still used today: ReLU instead of sigmoid/tanh, dropout to reduce overfitting, heavy data augmentation, and GPU parallelization
  - Showed CNNs excel at learning hierarchical visual features (edges → textures → object parts → full objects)
- Relevance to project: My choice of a convolutional neural network architecture is directly supported by this work — conv layers are highly effective for extracting spatial patterns (e.g., finger positions, hand shapes) from ASL images.  
  The simple CNN I built in Colab follows the same principles (stacked conv + pooling + dropout).  
  Future improvement: Consider more modern CNNs (e.g. MobileNet or ResNet) inspired by later papers that build on AlexNet's success.