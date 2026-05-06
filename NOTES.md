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
## 4/7 - 4/13
 
### 5-Class Specialist Model Results & Keras Version Fix
 
- 5-class specialist (A, M, N, O, T) trained successfully on Colab with ~3,300 images/class (3,000 Kaggle + 300 webcam)
- Model performed well on live webcam for those 5 letters — confirmed that webcam captures meaningfully close the domain gap even at only ~10% of training data
- A was occasionally confused unless hand was turned straight toward camera, but overall a positive sign
- **Critical issue encountered**: `.keras` model files from Colab would not load on local Mac due to Keras version mismatch
  - Colab runs Keras 3.x which includes `quantization_config` metadata in saved Dense layer configs
  - Local Mac's older Keras cannot deserialize this metadata — fails on both `.keras` and `.h5` formats
  - **Fix**: Save weights only with `model.save_weights()` in Colab, rebuild the identical architecture locally in the live script, then call `model.load_weights()`
  - Architecture in live script must EXACTLY match training (same layers, same sizes, same dropout, same number of fine-tune layers) or weight loading fails
  - This approach works because weights are just numpy arrays — no version-dependent metadata
- Also created the "demonstration" model: retrained the original bad model (black background dataset, 36 classes) with bugs fixed (no double preprocessing, correct image size) for side-by-side comparison during midterm demo
  - Saved as `asl_modelDemonstration_v2.weights.h5` using same weights-only approach
### Scaling to Full 27-Class Model
 
- Attempted full 27-class training at multiple resolutions:
  - 224×224: Crashed Colab Pro RAM repeatedly (with 3,000 images/class + copy step)
  - 224×224 with in-place trimming to 2,000/class: Still crashed
  - 224×224 with batch 32, shuffle 500, no copy: Still crashed
  - **160×160**: Successfully trained with in-place trimming, batch 32, shuffle 500
- Applied all optimizations from 5-class test: two-phase training, EarlyStopping, ReduceLROnPlateau
- Added confusion matrix to end of training script — prints per-class accuracy sorted worst-to-best and top 20 confusion pairs
- Validation accuracy: 99.7% (misleading — dominated by Kaggle images in validation set)
- Live performance: mediocre. Model collapsed to predicting N for most inputs, with correct letter sometimes appearing in top 3
### Domain Gap & Crop Improvements
 
- Identified domain gap as primary issue: Kaggle training images have generous framing with backgrounds; live webcam crops were tight (40px padding) and non-square, distorting hand shape during resize
- **Square crop fix**: Added logic to pad shorter dimension to match longer before resizing — preserves hand proportions
- **Increased padding**: 40px → 80px to include more context around hand, matching Kaggle image framing
- These changes immediately improved prediction variety in live inference
## 4/13 - 4/20
 
### 1,000 Webcam Image Capture Campaign
 
- Built custom capture script (`capture_training_data_224.py`) for systematic webcam data collection
  - Saves 224×224 square crops using same MediaPipe + square-pad logic as live inference
  - Crops taken BEFORE drawing landmarks so saved images are clean
  - Appends to existing data — safe to run multiple times without overwriting
  - Controls: SPACE=start/pause, TAB=next letter, A-Z=jump to letter, ESC=quit
  - Progress bar and time estimate displayed during capture
  - Counts existing images per letter and starts numbering from there
- Initially excluded J and Z as "motion letters" — later added them back after realizing Kaggle dataset has static frame images for both
- Captured 1,000 images per letter (26 letters, no space) at 224×224 resolution
  - Saved to `webcam_training_data_224_v21000images/`
  - Captured without band-aid (earlier 300-image set had band-aid on hand which could bias model)
  - Varied hand position: centered, edges of frame, close/far, slight wrist rotation
  - Captured near presentation building — whiteboard background with natural window light
### 224×224 Training with Memory Optimizations
 
- Applied all memory optimizations to enable 224×224 training on Colab Pro:
  - **Mixed precision** (`mixed_float16`): Stores activations/gradients in 16-bit, nearly halving memory
  - **Batch size 16** (down from 32)
  - **Shuffle buffer 256** (down from 1000)
  - **No `.cache()`**: Images load from disk per epoch instead of filling RAM
  - **In-place merge**: Webcam images copied directly into Kaggle folders — no dataset duplication
  - **Fine-tune 20 layers** (down from 30) to reduce gradient memory
  - Final Dense layer explicitly `dtype='float32'` for softmax numerical stability with mixed precision
- Full 3,000 Kaggle images/class retained (no trimming) + 1,000 webcam images merged in
- Training completed both phases (50 frozen + 25 fine-tune epochs) without crashing
- Phase 2 used all 25 epochs without EarlyStopping triggering — model kept improving throughout
### Two-Stage Model System (live_asl5.py)
 
- Combined the 27-class main model with the 5-class specialist into a single live script
- **How it works**:
  - Model 1 (27-class, 224×224) predicts on every frame
  - If prediction is A, M, N, O, or T AND confidence < 85%, Model 2 (5-class specialist, 160×160) is consulted
  - If specialist is more confident, its prediction overrides Model 1
  - For all other letters or when Model 1 is confident, Model 2 never runs
  - Display shows `[Model 1]` or `[Model 2]` label so user can see which model made the call
- **Consecutive agreement smoothing**: Display only updates after 6 matching predictions in a row (`AGREE_COUNT = 6`)
  - Prevents flickering from noisy frame-by-frame predictions
  - Makes predictions stable and readable during demo
  - Can be adjusted: lower = more responsive but jumpier, higher = more stable but slower to change
### Live Performance Assessment
 
- Live performance significantly improved from pre-webcam-capture baseline
- Validation accuracy: 99.7% (all 27 classes above 90%)
- Top confusions from confusion matrix: N→M (8), I→J (7), W→V (5), E→I (4), V→U (4)
- Live problem letters: R, T, P, O (confused with Y), W, D, S, E
- Discrepancy between 99.7% validation and mediocre live performance confirmed the domain gap issue — validation is ~80% Kaggle images, so EarlyStopping optimizes for Kaggle, not webcam
- Performance varies significantly with lighting conditions — model struggles in different environments than where webcam images were captured
## 4/20 - 4/22
 
### Upgrade Planning & Knowledge Transfer
 
- Developed comprehensive upgrade plan with tiered improvements:
  - **Tier 1 (High impact, easy)**: 256×256 resolution (professor recommended multiples of 8 for GPU alignment), webcam-only validation, label smoothing, AdamW optimizer, cosine annealing LR
  - **Tier 2 (High impact, moderate)**: EfficientNetB2 backbone, 40-50 fine-tune layers, 2,000-3,000 webcam images/class across 4-5 locations, test-time augmentation
  - **Tier 3 (Medium impact, easy)**: Unfreeze BatchNorm in Phase 2, class weights for confusable letters, LR warmup, increase dropout to 0.4
  - **Tier 4 (Medium impact, more effort)**: Model ensemble, CutMix/MixUp augmentation, multi-signer captures
- Professor feedback on mid-project poster: improve contrast/readability, streamline demo, update README with architecture details
- Created complete project knowledge transfer document for continuing work in new conversation — includes full bug history, file inventory, current code, architecture details, and upgrade plan
- Created research poster (42×34" format) with diagrams: inference pipeline flow, two-stage classification decision chart, training pipeline visualization, model evolution timeline
- Professor feedback on poster: remove tech from overview, just list tech stack, more detail on model architecture, combine preprocessing in pipeline, shrink acknowledgements, bigger more readable text
- Iterated on poster through v4 addressing all feedback — expanded Model Architecture section with four subsections (Why Transfer Learning, Backbone, Classification Head, Built-In Preprocessing)
## 4/22 - 4/29
 
### Comprehensive Code Review & Pipeline Rewrite
 
- Did a full code review of the existing training, inference, and capture scripts. Found several issues that
  collectively were responsible for the v4 99.7% validation accuracy not translating to good live performance:
  - **`RandomFlip("horizontal")` in augmentation was actively poisoning labels.** ASL signs are hand-specific —
    flipping G horizontally produces something resembling H. Including RandomFlip meant ~50% of training images
    were labeled wrong. Removed entirely; relying on natural hand-orientation variance instead.
  - **Random 80/20 validation split was leaking adjacent webcam frames.** Captures happen at ~6.7 fps, so
    frames 0.15s apart are nearly identical. Random split puts frame N in train and frame N+1 in val — model
    memorizes specific frames rather than generalizing. This explained the 99.7% validation vs mediocre live gap.
  - **`model.predict()` in the live loop has graph-rebuild overhead per call.** Designed for batch inference,
    not single-frame realtime. Switched to `model(x, training=False)` for direct forward-pass calls — 3-8×
    faster on M1, which is what makes 30 FPS achievable without a GPU.
  - **Square-crop bug in capture vs inference.** Edge-case hands near frame borders were silently stretched at
    training time but zero-padded at inference. Built a single shared `square_crop_with_padding` function used
    identically by capture, training, and live scripts to guarantee distribution consistency.
  - **Two-stage specialist had structural confidence bias.** A 5-class softmax peaks higher than a 27-class
    softmax just by having fewer classes to spread probability over, so the specialist almost always "won" the
    confidence comparison even when it was wrong. Scrapped the specialist; single-model architecture is simpler
    and the data improvements made it redundant.
  - **Class imbalance from 3:1 Kaggle:webcam ratio.** Phase 1 was dominated by Kaggle images. Phase 2 is now
    webcam-only to fix this without reducing Phase 1 volume.
  - **Shuffle buffer applied after batching** in some code paths — was shuffling batches not samples.
### Architectural Decisions for the Final Iteration
 
- **Kept MobileNetV2 over EfficientNetB2.** Rationale: ~3× faster on M1 inference, accuracy isn't the
  bottleneck for this task — distribution shift is. Speed-accuracy tradeoff strongly favors MobileNetV2.
- **Bumped input size 224×224 → 256×256.** 256 is divisible by MobileNetV2's stride of 32 (256/32 = 8 clean),
  produces a clean 8×8 final feature map, slight accuracy gain for modest compute cost.
- **Skipped TPU.** TPUStrategy rewrite wasn't worth it; A100 on Colab Pro+ is faster for this model anyway.
- **Two-phase training with split data sources.**
  - Phase 1: frozen backbone, Kaggle + webcam combined (~163k images) — head learns 27-class discrimination
    with maximum diversity
  - Phase 2: top 40 layers unfrozen, webcam-only (~82k images) — backbone adapts to deployment distribution
    without being pulled back toward Kaggle-style images
- **AdamW + CosineDecay + label smoothing 0.1.** AdamW handles weight decay correctly (Adam's
  weight_decay parameter modifies the gradient instead, which interacts badly with adaptive LR). CosineDecay
  is a deterministic schedule — doesn't depend on possibly-leaky validation feedback like ReduceLROnPlateau
  does. Label smoothing is free regularization that prevents overconfidence.
- **Mixed precision (float16) with float32 master weights** — ~2× faster on A100 with no accuracy loss.
- **BatchNorm kept in inference mode during fine-tuning** (`base(x, training=False)`). With small batches and
  the relatively small Phase 2 dataset, training-mode BN statistics are noisy. Inference-mode uses reliable
  population statistics from pretraining. This is the recommended Keras transfer learning pattern.
### Three Production Scripts (Final iteration)
 
- **`train_asl_Final.py`** — split-phase training with session-separated validation
- **`live_asl_Final.py`** — single-model inference, square_crop_with_padding, 6-frame agreement smoothing,
  `model(x, training=False)` for speed
- **`capture_training_data_Final.py`** — 256×256 captures with identical crop function as live script,
  `SKIP_PADDED_CROPS=True` flag (refuses to save edge-case captures with black bars), 2000 captures per
  letter cap, append-mode safe across multiple sessions
### Multi-Session Data Capture (~3,050 imgs/letter)
 
- Captured ~3,050 images per letter across **6 distinct conditions** in the demo building, deliberately chosen
  to break shortcut learning:
  1. **Main area** — natural top + window light, white wall background
  2. **Dim room** — blank wall, low ambient light
  3. **Whiteboard** — with handwritten content (lines, shapes)
  4. **Wood panel** — textured non-blank surface vs. white-wall bias
  5. **Bookshelf** — cluttered with chair, books, frames
  6. **Yellow wall** — different room, color-cast lighting
- Decisions made mid-capture:
  - Brown sweatshirt mid-session was OK for most letters — concerns about M/N (loose fabric in lower frame
    making fist letters look different) addressed by raising hand position so cuff doesn't dominate.
  - Bandaid on cut knuckle was non-issue — hand shape is what matters, not skin texture.
  - **Letter-correlated background area flagged as real concern.** For fist letters (M, N) the wrist angles
    toward camera, exposing more sweater area; for open-hand letters (B, L) the wrist faces away. That's a
    real correlation the model could exploit. Solution: keep clothing constant within a session, vary across
    sessions, raise hand position.
  - Skipped sunny-day captures due to time. ~70 minutes of additional capture beyond initial sessions.
- Class imbalance: 'space' was at ~1300 mid-capture, captured 1300 more to bring to ~2600 (still slightly
  under others but workable). Letter X ended up at 3500 (extra captures from earlier session before class
  cap was enforced).
### Final Training Run Results
 
- Trained on Colab Pro+ A100. Total time: **~70 minutes** (vs 17 hours on TPU v6 for v4). Better data, less
  compute, more honest result.
- Phase 1 (frozen backbone, Kaggle + webcam, ~163k images):
  - 30 epochs max, EarlyStopping at epoch 12, best at epoch 5
  - Final: **84.2% val accuracy**
- Phase 2 (top 40 layers unfrozen, webcam-only, ~82k images):
  - 20 epochs max, EarlyStopping at epoch 20, best at epoch 15
  - Final: **98.3% val accuracy** on session-separated validation set (6,830 images held out from a separate
    capture session on a different day in a different room)
- Per-class breakdown:
  - 21 of 27 letters at ≥99% accuracy
  - 7 letters at perfect 100%: B, C, F, G, H, J, L, M, P, Q, S, T, W, space (some shared)
  - **Letter E at 78.7%** — all 54 errors were E predicted as S (closed-fist visual ambiguity, both signs
    differ only in subtle thumb/finger placement)
  - Other notable: I→J (13), X→S (8), D→O (7), U→H (7)
- Phase 2 fine-tuning lifted accuracy from 84.2% → 98.3% — strong evidence the split-phase approach is
  worth the complexity. Adapting the backbone to deployment distribution is responsible for ~14 percentage
  points of the gap.
- Validation methodology: **session-separated** (not random-split). Held-out set captured on a different day,
  in a different room, with different backgrounds. The 98.3% is honest — it predicts deployment performance,
  unlike v4's 99.7% which was inflated by random-split leakage.
## 4/29 - 5/2
