# Real-Time ASL Fingerspelling Recognition
 
A live American Sign Language fingerspelling recognizer that runs on a standard laptop webcam at 30 FPS without a dedicated GPU. The system recognizes 27 classes (A–Z plus space) using MediaPipe for hand detection and a fine-tuned MobileNetV2 classifier. Achieves **98.3% accuracy on session-separated validation** — accuracy measured on a separate capture session in a different room than the training data.
 
**CPS 485 Senior Capstone — SUNY New Paltz — Spring 2026**
**Author:** AJ Zippo
**Advisor:** Dr. Curry
 
---
 
## How It Works
 
Each webcam frame goes through five stages in about 33 milliseconds:
 
1. OpenCV reads a frame from the webcam.
2. MediaPipe finds the hand and returns 21 landmark points. If no hand is in frame, the classifier doesn't run.
3. The script crops a square region around the hand, padding with black if the hand is near a frame edge, and resizes to 256×256.
4. The crop is fed through a MobileNetV2 backbone (pretrained on ImageNet, top 40 layers fine-tuned on ASL data) followed by a small classifier head: Dense 512 → Dense 256 → Dense 27 with softmax. Output is a probability for each of the 27 letters.
5. The displayed prediction updates only when 6 consecutive frames agree on the same letter, which keeps the on-screen text from flickering.
---
 
## Requirements
 
- **Python 3.9** (other versions may work but are untested)
- **macOS** with M1/M2/M3 chip recommended (the live script uses `model(x, training=False)` for fast single-frame inference on Apple Silicon)
- A webcam
Python dependencies are listed in `requirements.txt`:
 
```
tensorflow-macos
tensorflow-metal
mediapipe
opencv-python
numpy
```
 
---
 
## Installation
 
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CPS485project.git
cd CPS485project
 
# 2. Create and activate a virtual environment
python3.9 -m venv asl_env
source asl_env/bin/activate
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
---
 
## Running the Live Demo
 
Once dependencies are installed:
 
```bash
python live_asl_Final.py
```
 
A webcam window will open. Sign letters with one hand in front of the camera. The predicted letter, confidence percentage, and top 3 candidates appear overlaid on the video feed.
 
**Controls:**
- Press `q` to quit
**What to expect:**
- The displayed letter updates only after 6 consecutive frames agree, so brief signs may not register — hold the sign for ~200ms.
- Letters J and Z (motion-defined in ASL) are recognized weakly because this is a single-frame CNN. Working sign and good lighting help.
- Letter E is the weakest static letter (78.7%) because it visually resembles S. Both are closed-fist shapes.
**Required files** (already in the repo):
- `live_asl_Final.py` — the inference script
- `asl_model_Final.weights.h5` — trained model weights
- `class_names_Final.json` — class label ordering
All three must be in the same directory.
 
---
 
## Retraining the Model
 
To retrain on Google Colab (free tier works, but Colab Pro+ with an A100 GPU is recommended for ~70 minute training time vs many hours on free tier):
 
1. **Download the Kaggle ASL Alphabet dataset.** Available at: https://www.kaggle.com/datasets/grassknoted/asl-alphabet — rename to `archive (1).zip` (this is the filename the notebook expects).
2. **Prepare webcam capture zips.** You'll need:
   - `webcam_training_data_Final.zip` — your own webcam captures, ideally captured across multiple lighting/background conditions (see *Capturing New Training Data* below)
   - `webcam_val_Final.zip` — a *separate* session captured on a different day in a different location for honest validation accuracy
3. **Open `train_asl_Final.ipynb` in Google Colab.**
4. **Upload all three zip files** to the Colab session via the file browser.
5. **Run cells in order:**
   - Cell 1 (Cleanup): clears leftover folders from previous runs
   - Cell 3 (Install): `pip install tensorflow`
   - Cell 6 (Training): runs the full split-phase training pipeline
     - Phase 1: trains the classifier head on combined Kaggle + webcam (~163k images)
     - Phase 2: fine-tunes the top 40 backbone layers on webcam-only (~82k images)
     - Outputs: `asl_model_Final.weights.h5` and `class_names_Final.json`
6. **Download the two output files** from Colab and place them in your local repo root alongside `live_asl_Final.py`.
**Expected training time:** ~70 minutes on Colab Pro+ A100. ~3-5 hours on free-tier Colab.
 
**Expected final accuracy:** 96-99% on session-separated validation, depending on the diversity of your capture conditions.
 
---
 
## Capturing New Training Data
 
The capture script saves webcam frames into per-letter folders, using the same square-crop logic as the live inference script (so train and inference see the same image distribution).
 
```bash
python capture_training_data_Final.py
```
 
**Controls:**
- `SPACE` — start/pause capture
- `TAB` — advance to next letter
- `A`-`Z` — jump to a specific letter
- `ESC` — quit
The script appends to existing folders, so it's safe to run multiple times across multiple sessions. **For best results, capture across multiple distinct lighting and background conditions** — see the `sample_capture_conditions/` folder for examples of the six conditions used in the Final training data.
 
---
 
## Results
 
| Metric | Value |
|---|---|
| Validation accuracy (session-separated) | **98.3%** |
| Inference speed | **30 FPS** on MacBook M1, no GPU |
| Classes | 27 (A–Z + space) |
| Input resolution | 256 × 256 |
| Letters above 99% individual accuracy | 21 of 27 |
| Letters at perfect 100% | 7 |
| Weakest letter | E at 78.7% (visually similar to S) |
| Training time (Colab A100) | ~70 minutes |
 
**Phase 1** (frozen backbone, Kaggle + webcam combined): 84.2% val accuracy.
**Phase 2** (top 40 layers fine-tuned, webcam-only): **98.3%** val accuracy.
 
---
 
## Known Limitations
 
- **Letters J and Z** are recognized weakly. Both are defined by hand motion in ASL, and a single-frame CNN cannot perceive motion. Fix would require a temporal model (LSTM or Transformer over a sequence of frames).
- **Letter E ↔ S confusion.** Both are closed-fist hand shapes differing only in subtle thumb placement. All 54 of E's validation errors were predicted as S. Targeted capture from angles emphasizing the distinguishing feature would help.
- **Cross-person generalization is unmeasured.** The model was trained primarily on one person's hands. Performance on different hand sizes, skin tones, or signing styles is unknown.
- **Performance varies with environment.** The model was trained on six specific lighting/background conditions in the demo building. Drastically different conditions (extreme low light, outdoor sun, unusual backgrounds) may degrade accuracy.
---
 
## Repository Structure
 
```
CPS485project/
├── README.md                          This file
├── NOTES.md                           Full iteration log from project start
├── requirements.txt                   Python dependencies
├── .gitignore                         Files excluded from version control
│
├── live_asl_Final.py                  Live webcam inference (run this for demo)
├── capture_training_data_Final.py     Capture new webcam training data
├── train_asl_Final.ipynb              Colab training notebook
│
├── asl_model_Final.weights.h5         Trained model weights (52 MB)
├── class_names_Final.json             Class label ordering (required for inference)
│
├── poster/                            Final research poster
│   └── ASL_Research_Poster_Finalone.pptx
│
├── sample_capture_conditions/         6 representative images showing capture diversity
│   ├── 01_main_area_O.jpg
│   ├── 02_dim_room_O.jpg
│   ├── 03_whiteboard_O.jpg
│   ├── 04_wood_panel_O.jpg
│   ├── 05_bookshelf_O.jpg
│   └── 06_yellow_wall_O.jpg
│
└── _archive/                          Historical work (preserved but not part of final deliverable)
    ├── README.md
    ├── old_inference_scripts/         Previous live_asl*.py iterations (v1-v7)
    ├── old_capture_scripts/           Earlier capture script versions
    ├── old_models/                    Trained weights from earlier iterations
    │   └── pre_v3_36class_era/        Earliest models from Feb-Mar (36 classes)
    ├── old_notebooks/                 Earlier training/test notebooks
    └── project_brief.pdf              Original assignment description
```
 
---
 
## Iteration History
 
This project went through six major iterations before reaching the Final version. Each version corrected specific bugs or limitations from the previous one — including a preprocessing collapse that took weeks to debug, a validation leakage issue that inflated v4's reported accuracy from a real 98.3% to a misleading 99.7%, and a two-stage specialist architecture that was eventually scrapped in favor of a single MobileNetV2.
 
Full chronological notes from 2/3/2026 through the Final iteration are in [`NOTES.md`](NOTES.md).
 
---
 
## References
 
1. Howard et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
2. Zhang et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking.* CVPR Workshop.
3. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization* (AdamW). ICLR.
4. Szegedy et al. (2016). *Rethinking the Inception Architecture for Computer Vision* (Label Smoothing). CVPR.
5. Kaggle: grassknoted/asl-alphabet. *ASL Alphabet Image Dataset.*
---
 
## Acknowledgements
 
Developed for CPS 485 Senior Capstone, SUNY New Paltz Computer Science Department. Thanks to my advisor Dr. Curry for guidance throughout the iteration cycle, the Kaggle community for the public ASL Alphabet dataset, and the open-source maintainers of TensorFlow, MediaPipe, and OpenCV.