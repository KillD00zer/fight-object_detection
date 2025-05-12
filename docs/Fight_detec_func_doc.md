# Documentation: `Fight_detec_func.py`

## Overview

This script provides a function, `fight_detec`, designed to analyze a video file and predict whether it contains fight-like activity or normal behavior. It utilizes a pre-trained deep learning model (specifically, a Keras/TensorFlow model) to make this classification. The script handles loading the model, extracting relevant frames from the video using the `frame_slicer` module, performing the prediction, and calculating a confidence score. It also includes optional debugging features like printing status messages and saving visualizations.

## Functionality

The script operates primarily through the `fight_detec` function, which internally uses a helper class `FightDetector`. The process involves:

1.  **Configuration:** Defines constants for the model path (`MODEL_PATH`), the number of frames required by the model (`N_FRAMES`), the expected image size (`IMG_SIZE`), and the directory for saving results (`RESULT_PATH`).
2.  **`FightDetector` Class Initialization:** When `fight_detec` is called, it creates an instance of `FightDetector`. The class constructor (`__init__`) immediately attempts to load the pre-trained model using the `_load_model` method.
3.  **Model Loading (`_load_model`):** Loads the Keras model specified by `MODEL_PATH`. It includes error handling for loading failures. If `debug` mode is enabled, it prints the model's input shape upon successful loading.
4.  **Frame Extraction (`_extract_frames`):** This method is called by `predict`. It uses the `extract_video_frames` function (imported from `frame_slicer.py`) to get `N_FRAMES` frames, each resized to `IMG_SIZE`, from the input `video_path`. It handles potential errors during frame extraction. In debug mode, it checks for blank frames and saves a sample extracted frame (`debug_frame.jpg`) to the `RESULT_PATH`.
5.  **Prediction (`predict`):** This is the core method of the `FightDetector` class.
    *   **Input Validation:** Checks if the `video_path` exists.
    *   **Frame Extraction:** Calls `_extract_frames` to get the video frames.
    *   **Frame Validation:** Checks if frame extraction was successful, if the correct number of frames (`N_FRAMES`) was returned, and if the frames are not all blank.
    *   **Model Inference:** Prepares the frames array (adds a batch dimension) and feeds it into the loaded `model` using `model.predict()`.
    *   **Result Interpretation:** Interprets the model's raw output score. A score above a threshold (0.61 in this script) is classified as "FIGHT", otherwise "NORMAL".
    *   **Confidence Calculation:** Calculates a heuristic confidence score based on how far the prediction score is from the threshold, clamped between 0 and 100%.
    *   **Debugging:** If `debug` mode is on, it calls `_debug_visualization`.
    *   **Return Value:** Returns a formatted string (e.g., "FIGHT (85.2% confidence)") and the raw prediction score. Handles potential exceptions during prediction.
6.  **Debug Visualization (`_debug_visualization`):** If `debug` mode is enabled, this method generates a plot using Matplotlib showing the first few extracted frames, the prediction score, and the final result. This plot is saved as a PNG image in the `RESULT_PATH`, named after the input video.
7.  **`fight_detec` Function Execution:** The main function orchestrates the process by creating the `FightDetector` instance and calling its `predict` method. It handles the case where model loading might have failed during initialization.

## Configuration Constants

*   `MODEL_PATH` (str): The absolute path to the saved Keras model file (`.h5`). **Crucial:** This path must be correct for the script to function.
*   `N_FRAMES` (int): The number of frames the model expects as input (e.g., 30). This must match the input shape the model was trained with.
*   `IMG_SIZE` (tuple): The spatial dimensions (width, height) the model expects for each frame (e.g., (96, 96)). This must also match the model's training configuration.
*   `RESULT_PATH` (str): The absolute path to the directory where debug outputs (sample frame, visualization plots) will be saved.

## Function Signature (`fight_detec`)

```python
def fight_detec(video_path: str, debug: bool = True) -> tuple[str, float | None]:
```

### Parameters:

*   `video_path` (str): The file path to the input video to be analyzed.
*   `debug` (bool, optional): If `True`, enables printing detailed logs and saving visualization images. Defaults to `True`.

### Returns:

*   `tuple`: A tuple containing:
    *   `str`: A string describing the prediction result (e.g., "FIGHT (92.1% confidence)", "NORMAL (75.5% confidence)") or an error message (e.g., "Error: Video not found", "Error: Model loading failed").
    *   `float | None`: The raw numerical prediction score from the model (typically between 0 and 1). Returns `None` if an error occurred before prediction.

## Usage Example

```python
from Fight_detec_func import fight_detec
import os

video_to_analyze = r"D:\K_REPO\ComV\try\some_video.mp4" # Use raw string for Windows paths

if os.path.exists(video_to_analyze):
    result_string, score = fight_detec(video_to_analyze, debug=True)
    print(f"Video: {os.path.basename(video_to_analyze)}")
    print(f"Result: {result_string}")
    if score is not None:
        print(f"Raw Score: {score:.4f}")
else:
    print(f"Error: Input video not found at {video_to_analyze}")

```

## Dependencies

*   **TensorFlow (`tensorflow`):** Used for loading and running the deep learning model. Install via `pip install tensorflow`.
*   **OpenCV (`cv2`):** Used indirectly via `frame_slicer` and potentially for saving debug frames. Install via `pip install opencv-python`.
*   **NumPy (`numpy`):** Used for numerical operations, especially handling frame arrays. Install via `pip install numpy`.
*   **Matplotlib (`matplotlib`):** Used for generating debug visualizations. Install via `pip install matplotlib`.
*   **`frame_slicer.py`:** A custom module (expected to be in the same directory or Python path) containing the `extract_video_frames` function.
