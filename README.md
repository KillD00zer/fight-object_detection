# Video Analysis Project: Fight and Object Detection

## 1. Overview

This project analyzes video files to perform two main tasks:
*   **Fight Detection:** Classifies video segments as containing a "FIGHT" or being "NORMAL" using a custom-trained 3D Convolutional Neural Network (CNN).
*   **Object Detection:** Identifies and tracks specific objects within the video using a pre-trained YOLOv8 model.

The system processes an input video and outputs the fight classification result along with an annotated version of the video highlighting detected objects.

## 2. Features

*   Dual analysis: Combines action recognition (fight detection) and object detection.
*   Custom-trained model for fight detection tailored to specific data.
*   Utilizes state-of-the-art YOLOv8 for object detection.
*   Generates an annotated output video showing detected objects and their tracks.
*   Provides confidence scores for fight detection results.
*   Includes scripts for both inference (`full_project.py`) and training (`trainig.py`) the fight detection model.

## 3. Project Structure

```
ComV/
├── [Project Directory]/        # e.g., AI_made
│   ├── full_project.py         # Main script for running inference
│   ├── Fight_detec_func.py     # Fight detection logic and model loading
│   ├── objec_detect_yolo.py    # Object detection logic using YOLOv8
│   ├── frame_slicer.py         # Utility for extracting frames for fight detection
│   ├── trainig.py              # Script for training the fight detection model
│   ├── README.md               # This documentation file
│   └── trainnig_output/        # Directory for training artifacts
│       ├── final_model_2.h5    # Trained fight detection model (relative path)
│       └── checkpoint/         # Checkpoints saved during training (relative path)
│       └── training_log.csv    # Log file for training history (relative path)
│   └── yolo/                   # (Assumed location)
│       └── yolo/
│           └── best.pt         # Pre-trained YOLOv8 model weights (relative path)
├── train/
│   ├── Fighting/               # Directory containing fight video examples (relative path)
│   └── Normal/                 # Directory containing normal video examples (relative path)
└── try/
    ├── result/                 # Directory where output videos are saved (relative path)
    └── ... (Input video files) # Location for input videos (example)
```

*(Note: Model paths and data directories might be hardcoded in the scripts. Consider making these configurable or using relative paths.)*

## 4. Setup and Installation

**Python Version:**

*   This project was developed and tested using Python 3.10.

**Dependencies:**

Based on the code imports and `pip freeze` output, the following libraries and versions were used:

*   `opencv-python==4.11.0.86` (cv2)
*   `numpy==1.26.4`
*   `tensorflow==2.19.0` (tf)
*   `ultralytics==8.3.108` (for YOLOv8)
*   `matplotlib==3.10.1` (for debug visualizations)
*   `scikit-learn==1.6.1` (sklearn - used in `trainig.py`)

*(Note: Other versions might also work, but these are the ones confirmed in the development environment.)*

**Installation (using pip):**

```bash
pip install opencv-python numpy tensorflow ultralytics matplotlib scikit-learn
```

**Models:**

1.  **Fight Detection Model:** Ensure the trained model (`final_model_2.h5`) is present in the `trainnig_output` subdirectory relative to the script location.
2.  **YOLOv8 Model:** Ensure the YOLO model (`best.pt`) is present in the `yolo/yolo` subdirectory relative to the script location.

*(Note: Absolute paths might be hardcoded in the scripts and may need adjustment depending on the deployment environment.)*

## 5. Usage

To run the analysis on a video file:

1.  Navigate to the `d:/K_REPO/ComV/AI_made/` directory in your terminal (or ensure Python's working directory is `d:/K_REPO`).
2.  Run the main script:
    ```bash
    python full_project.py
    ```
3.  The script will prompt you to enter the path to the video file:
    ```
    Enter the local path : <your_video_path.mp4>
    ```
    *(Ensure you provide the full path, potentially removing extra quotes if copying from Windows Explorer.)*

**Output:**

*   The console will print the fight detection result (e.g., "FIGHT (85.3% confidence)") and information about the object detection process.
*   An annotated video file will be saved in the `D:\K_REPO\ComV\try\result` directory. The filename will include the original video name and the unique detected object labels (e.g., `input_video_label1_label2_output.mp4`).
*   If debug mode is enabled in `Fight_detec_func.py`, additional debug images might be saved in the result directory.

## 6. Module Descriptions

*   **`full_project.py`:** Orchestrates the process by taking user input and calling the fight detection and object detection functions.
*   **`Fight_detec_func.py`:**
    *   Contains the `fight_detec` function and `FightDetector` class.
    *   Loads the Keras model (`final_model_2.h5`).
    *   Uses `frame_slicer` to prepare input for the model.
    *   Performs prediction and calculates confidence.
    *   Handles debug visualizations.
*   **`objec_detect_yolo.py`:**
    *   Contains the `detection` function.
    *   Loads the YOLOv8 model (`best.pt`).
    *   Iterates through video frames, performs object detection and tracking.
    *   Generates and saves the annotated output video.
    *   Returns detected object labels.
*   **`frame_slicer.py`:**
    *   Contains the `extract_video_frames` utility function.
    *   Extracts a fixed number of frames, resizes, normalizes, and handles potential errors during extraction.
*   **`trainig.py`:**
    *   Script for training the fight detection model.
    *   Includes `VideoDataGenerator` for loading/processing video data.
    *   Defines the 3D CNN model architecture.
    *   Handles data loading, splitting, training loops, checkpointing, and saving the final model.

## 7. Training Data

### Dataset Composition
| Category       | Count | Percentage | Formats       | Avg Duration |
|----------------|-------|------------|---------------|--------------|
| Fight Videos   | 2,340 | 61.9%      | .mp4, .mpeg   | 5.2 sec      |
| Normal Videos  | 1,441 | 38.1%      | .mp4, .mpeg   | 6.1 sec      |
| **Total**      | **3,781** | **100%**  |               |              |

### Technical Specifications
- **Resolution:** 64×64 pixels
- **Color Space:** RGB
- **Frame Rate:** 30 FPS (average)
- **Frame Sampling:** 50 frames per video
- **Input Shape:** (30, 96, 96, 3) - Model resizes input

### Data Sources
- Fighting videos: Collected from public surveillance datasets
- Normal videos: Sampled from public CCTV footage
- Manually verified and labeled by domain experts

### Preprocessing
1. Frame extraction at 30 frames/video
2. Resizing to 96×96 pixels
3. Normalization (pixel values [0,1])
4. Temporal sampling to 30 frames for model input

## 8. Models Used

*   **Fight Detection:** A custom 3D CNN trained using `trainig.py`. Located at `D......final_model_2.h5`. Input shape expects `(30, 96, 96, 3)` frames.
*   **Object Detection:** YOLOv8 model. Weights located at `D:\K_REPO\ComV\yolo\yolo\best.pt`. This model is trained to detect the following classes: `['Fire', 'Gun', 'License_Plate', 'Smoke', 'knife']`.

## 7a. Fight Detection Model Performance

The following metrics represent the performance achieved during the training of the `final_model_2.h5`:

*   **Best Training Accuracy:** 0.8583 (Epoch 7)
*   **Best Validation Accuracy:** 0.9167 (Epoch 10)
*   **Lowest Training Loss:** 0.3636 (Epoch 7)
*   **Lowest Validation Loss:** 0.2805 (Epoch 8)

*(Note: These metrics are based on the training run that produced the saved model. Performance may vary slightly on different datasets or during retraining.)*

## 8. Configuration

Key parameters and paths are mostly hardcoded within the scripts:

*   `Fight_detec_func.py`: `MODEL_PATH`, `N_FRAMES`, `IMG_SIZE`, `RESULT_PATH`.
*   `objec_detect_yolo.py`: YOLO model path, output directory path (`output_dir`), confidence threshold (`conf=0.7`).
*   `trainig.py`: `DATA_DIR`, `N_FRAMES`, `IMG_SIZE`, `EPOCHS`, `BATCH_SIZE`, `CHECKPOINT_DIR`, `OUTPUT_PATH`.

*Recommendation: Refactor these hardcoded values into a separate configuration file (e.g., YAML or JSON) or use command-line arguments for better flexibility.*

## 9. Training the Fight Detection Model

To retrain or train the fight detection model:

1.  **Prepare Data:** Place training videos into `D:\K_REPO\ComV\train\Fighting` and `D:\K_REPO\ComV\train\Normal` subdirectories.
2.  **Run Training Script:** Execute `trainig.py`:
    ```bash
    python trainig.py
    ```
3.  The script will load data, build the model (or resume from a checkpoint if `RESUME_TRAINING=1` and a checkpoint exists), train it, and save the final model to `D:\K_REPO\ComV\AI_made\trainnig_output\final_model_2.h5`. Checkpoints and logs are saved in the `trainnig_output` directory.
