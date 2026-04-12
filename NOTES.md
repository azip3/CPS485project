# Project Notes

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

## 2/10 - 2/17

## Context
-**37 Classes**: 37th class was removed since it was a copy of the dataset inside a folder of the same name. This 37th class confused model while training since it made no sense that 2 parallel data sets exist.

## Overview
- **Goal**: Real-time American Sign Language  recognition from webcam using hand landmark detection (MediaPipe) + CNN classification (36 classes: 0-9 + a-z).
- **Dataset**: Custom `asl_dataset` folder with 36 class subdirectories, loaded via `tf.keras.utils.image_dataset_from_directory` (224×224 input size, 20% validation split).
- **Baseline**: Custom from-scratch CNN (3 conv blocks: 32→64→128 filters, dropout 0.6) → ~75% validation accuracy after fixes and training.
- **Approach**: Switched to transfer learning using pre-trained ImageNet models to boost accuracy and generalization.

## MobileNetV2 (Lightweight Transfer Learning Model)
- **Why chosen**: Fast inference for real-time webcam demo, low memory footprint, strong performance on small/medium datasets, widely used in ASL/sign language projects.
- **Architecture**:
  - Base: MobileNetV2 (frozen initially, then fine-tuned last half of layers)
  - Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.4) → Dense(36, softmax)
  - Input size: 224×224 (resized from cropped hand region)
  - Preprocessing: `applications.mobilenet_v2.preprocess_input`
  - Optimizer: RMSprop (initial 0.001 → 1e-5 for fine-tuning)
- **Training**:
  - Phase 1 (frozen base, feature extraction): 20 epochs
  - Phase 2 (fine-tuning last half): 10 epochs
  - Augmentation: Horizontal flip, rotation ±27°, brightness/contrast ±20%, zoom ±15%, translation ±10%
- **Results**:
  - After Phase 1 (20 epochs): ~85% validation accuracy
  - After Phase 2 (fine-tuning, total 30 epochs): [ 85% accuracy]
  - Final validation accuracy: [92% accuracy]
- **Key observation**: Significant jump from baseline custom CNN (~75%) due to pre-trained ImageNet features + better generalization. Still room for improvement with more epochs, lower dropout, or Adam optimizer.

## ResNet50 (Deeper Transfer Learning Model – In Progress)
- **Why chosen**: Professor-suggested ; deeper residual architecture for potentially higher accuracy ceiling; good for comparison with mobilenet.
- **Architecture**:
  - Base: ResNet50 (frozen initially, then fine-tuned last half of layers)
  - Head: Same as MobileNetV2 (GlobalAveragePooling2D → Dense(128) → Dropout(0.4) → Dense(36, softmax))
  - Input size: 224×224
  - Preprocessing: `applications.resnet.preprocess_input`
  - Optimizer: RMSprop (same schedule as MobileNetV2)
- **Training status**:
  - Phase 1 (frozen base): [10% less accuracy by 4th epoch compared to mobilenet]
  - Phase 2 (fine-tuning): []
  - Augmentation: Identical to MobileNetV2 pipeline
- **Preliminary / Expected Results**:
  - Validation accuracy after Phase 1: []
  - Final expected range: 95–99% (based on similar ASL transfer learning benchmarks)
  - Inference speed: Slower than MobileNetV2 (expected 50–100 ms/frame → lower FPS in live demo)
- **Notes**: Will provide direct architecture & accuracy comparison once fully trained.

## 2/17 - 2/24

-**Debugging Domain Shift and Pipeline Corruption**
Diagnosing J-at-100% Confidence

live_asl.py predicted "J" at 100% confidence regardless of hand position
Added debug output: top-5 predictions + input value range
Debug showed all 36 classes scoring ~3–4% — statistically equivalent to random guessing (1/36 = 2.78%)

Root Cause 1 — Rescaling Bug Inside Model

Model architecture contained a Rescaling(255.0) layer inside the model itself
Caused training inputs to be in the range [0, 65025] instead of the correct [0, 255] for preprocess_input()
Model appeared to train normally (reaching 94% val accuracy) but learned from completely corrupted inputs
Fix: Removed Rescaling(255.0) layer entirely; full retraining from scratch required
- A related bug was also found in the background augmentation threshold: the mask was set
to `30.0/255.0 = 0.117` instead of `30.0`, meaning it never triggered because pixel
values were in [0, 255] not [0, 1] — the mask silently passed every pixel through
without augmenting anything

- This was confirmed when an 82% validation accuracy model — which appeared to be a clean
  retrain — was still predicting J at 100% in live inference. Investigation revealed it had
  been trained with the `Rescaling(255.0)` bug present the entire time, meaning the 82%
  figure was meaningless. The model had to be discarded and retrained from scratch with the
  layer fully removed

Lesson: A model trained on a corrupted input distribution cannot be rescued by fixing inference — retraining is always required



Root Cause 2 — Domain Shift from Black Background Dataset

Training dataset had plain black backgrounds; real webcam input has real room backgrounds
Model learned background color as a feature rather than hand shape — completely useless on real webcam input
94% validation accuracy was meaningless because the validation set came from the same broken distribution
Fix: Migrate to a dataset with real-world backgrounds

Root Cause 3 — tf.py_function Pipeline Corruption

Background augmentation implemented by wrapping NumPy ops in tf.py_function inside a tf.data pipeline
Silently caused label/image misalignment without raising any errors
Fix: Rewrote using native TF ops (tf.reduce_all, tf.random.uniform); later removed entirely once the new dataset provided real backgrounds natively
Lesson: tf.py_function wrapping NumPy inside tf.data can silently corrupt the pipeline — always use native TF ops
## 2/24 - 3/3

-**Saving Models**
- Model initially saved as `BLANK.h5` then migrated to `.keras` format to resolve
  serialization issues with the legacy HDF5 format — this is also why the ResNet50 `.h5`
  file later caused import issues when attempting to load it for live inference.


-**Kaggle Dataset Migration and Image Size Decision**
Migrating to the Kaggle Large-Scale ASL Dataset

Identified grassknoted/asl-alphabet on Kaggle:

87,000 images, 1.1 GB, real-world backgrounds, ~3,000 images per class, 29 classes (A–Z + space + del + nothing)


Upgraded to Google Colab Pro for longer session limits and T4/V100 GPU access
Uploaded archive (1).zip to Colab; added shutil.rmtree() pre-unzip cleanup to handle leftover files from crashed sessions
Dataset extracted to asl_alphabet_train/asl_alphabet_train/ — path required adjustment from expected location

Removing Problem Classes

nothing class removed: Contained real room photos — exactly matched webcam input distribution, causing 100% confidence predictions on background with no hand present
del class removed: Not a standard ASL fingerspelling sign; not relevant to the project's classification goals

Image Size — Settling on 160×160

With the larger Kaggle dataset loaded, attempted to increase input size from 128×128 → 224×224 for better detail
224×224 caused consistent Colab RAM crashes even on Pro tier
Tested intermediate sizes; settled on 160×160 as the largest size that fit reliably in Colab Pro RAM
160×160 used for all subsequent training
Lesson: RAM constraints in cloud environments require empirical testing of input sizes — there is no universal safe value

RAM Management in Colab

Removed .cache() — was attempting to load the entire 87,000-image dataset into RAM
Reduced shuffle buffer to 200, parallel workers to 2
Used in-place image deletion rather than copying to a separate balanced directory

## 3/3 - 3/10
-**Class Balancing Fix**
J-Only Predictions (Again) After Dataset Switch

After removing nothing class, model collapsed to predicting "J" at 100%
Root cause: .take() sampling on alphabetically-ordered batches heavily over-represented early letters (G, J)

Per-Image Balanced Sampling — The Correct Fix

Implemented a balanced folder builder:

Shuffles images within each class folder using random.seed(42) for reproducibility
Keeps exactly IMAGES_PER_CLASS (2,000) images per class by deleting excess in-place
Guarantees equal representation at the image level before any tf.data loading


Why this works where batch shuffling didn't: tf.data batch-level shuffling operates after images are already loaded in alphabetical order — it cannot retroactively fix the imbalance introduced by .take(). Per-class image sampling at the filesystem level is the only reliable fix.
Lesson: True class balance requires per-image sampling from each class folder individually — batch-level operations after loading are insufficient



-**Final Training and Live Demo**
Final Training Configuration

Dataset: grassknoted/asl-alphabet trimmed to 2,000 images per class, 27 classes (A–Z + space), ~54,000 total
Input size: 160×160
Backbone: MobileNetV2 (ImageNet weights)
Head: GlobalAveragePooling2D → Dense(512, ReLU) → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → Dense(27, softmax)
Augmentation: RandomFlip (horizontal), RandomRotation (0.15), RandomZoom (0.2), RandomBrightness (0.3), RandomContrast (0.3), RandomTranslation (0.1, 0.1)
Phase 1: Adam(1e-3), up to 40 epochs, EarlyStopping(patience=7), ReduceLROnPlateau
Phase 2: Adam(1e-5), unfreeze top 30 backbone layers, up to 20 epochs, EarlyStopping(patience=5)
Saved as: asl_model3_v2.keras + class_names3.json

## 3/17 - 3/24
-**Training**
Learned that I did not need to train in color for CNN. Swapped to training in Greyscale so I can train more photos from dataset.


## 3/24 - 4/7
-**Training**
GreyScale was less optimal of a path than I thought. Only decreases load by 1/3 from the colors but comes with signifacnt losses in accuracy. 
Started training on self captured data. Captures 300 images of my own hand making signs and incorporated to 3000 training images from Kaggle. Ten percent of  data comes from my own which should be minimal amount to make difference in learning from model. Will Increase by 5-10% increments to evaluate benefits.
# ── April 7th Test Notes ──────────────────────────────────────────────
# Notebook:  April7smallDataTest.ipynb
# Model:     asl_model_5class.keras
# Classes:   class_names_5class.json
# Classes trained: A, M, N, O, T (5 classes only)
# Data: All Kaggle images per class (~3000) + 300 webcam images per class
# Purpose: Targeted test to improve accuracy on confusable closed-fist letters
# Use with: live_asl3.py (change MODEL_PATH and CLASSES_FILE)
# ──────────────────────────────────────────────────────────────────────