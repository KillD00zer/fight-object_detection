# Documentation: `frame_slicer.py`

## Overview

This script provides a function, `extract_video_frames`, designed to reliably extract a fixed number of frames from short video files (typically 2-10 seconds long). It's optimized for common video formats like MP4 and MPEG. The primary goal is to prepare video data for input into machine learning models, particularly those requiring a consistent sequence length and frame size, such as action recognition or video classification models.

## Functionality

The `extract_video_frames` function performs the following steps:

1.  **Video Loading:** Opens the specified video file using OpenCV (`cv2.VideoCapture`). It includes basic error handling to check if the video can be opened.
2.  **Video Validation:** Retrieves the total number of frames and frames per second (FPS). It performs a basic check to ensure the video is valid (has at least one frame and a positive FPS).
3.  **Adaptive Frame Selection:** Calculates how many frames to skip (`frame_step`) between selections. This step is adaptive, meaning it adjusts based on the total number of frames in the video to ensure the selected frames are spread out relatively evenly across the video's duration, regardless of its exact length. The goal is to capture representative moments from the entire video clip.
4.  **Frame Extraction Loop:** Iterates `n_frames` times (default is 30) to collect the desired number of frames.
    *   **Position Calculation:** For each iteration, it calculates the specific frame number (`pos`) to read, aiming for an even distribution across the video timeline.
    *   **Frame Reading:** Attempts to read the frame at the calculated position using `cap.set()` and `cap.read()`.
5.  **Error Handling (Frame Level):** If reading a specific frame fails (e.g., due to corruption or reaching the end unexpectedly):
    *   It first tries to reuse the `last_good_frame` successfully read.
    *   If no good frame has been read yet, it generates a placeholder frame (a light gray image) to maintain the sequence length. This ensures the function always returns an array of the expected shape.
6.  **Frame Preprocessing:** For each successfully read frame:
    *   **Resizing:** Resizes the frame to the target `frame_size` (default 96x96 pixels) using `cv2.resize()`.
    *   **Color Conversion:** Converts the frame's color space from BGR (OpenCV's default) to RGB using `cv2.cvtColor()`. RGB is more standard for many deep learning models.
    *   **Normalization:** Converts the frame's data type to 32-bit float and normalizes the pixel values to the range [0.0, 1.0] by dividing by 255.0. This is a common preprocessing step for neural networks.
7.  **Resource Cleanup:** Releases the video capture object (`cap.release()`) to free up system resources.
8.  **Output:** Returns the collected frames as a single NumPy array. The shape of the array will be `(n_frames, height, width, 3)`, where `height` and `width` correspond to the `frame_size`.

## Function Signature

```python
def extract_video_frames(video_path: str, n_frames: int = 30, frame_size: tuple = (96, 96)) -> np.ndarray | None:
```

### Parameters:

*   `video_path` (str): The file path to the input video.
*   `n_frames` (int, optional): The exact number of frames to extract. Defaults to 30.
*   `frame_size` (tuple, optional): The target size (width, height) to resize each extracted frame to. Defaults to (96, 96).

### Returns:

*   `np.ndarray`: A NumPy array containing the extracted and processed frames. The shape is `(n_frames, frame_size[1], frame_size[0], 3)`.
*   `None`: Returns `None` if the video cannot be opened or is considered invalid.

## Usage Example

```python
import numpy as np
from frame_slicer import extract_video_frames

video_file = "path/to/your/video.mp4"
num_frames_to_get = 30
target_dimensions = (96, 96)

frames_array = extract_video_frames(video_file, n_frames=num_frames_to_get, frame_size=target_dimensions)

if frames_array is not None:
    print(f"Successfully extracted frames. Array shape: {frames_array.shape}")
    # Now 'frames_array' can be used as input for a model
else:
    print(f"Failed to extract frames from {video_file}")

```

## Dependencies

*   **OpenCV (`cv2`):** Used for video loading, frame reading, resizing, and color conversion. Install via `pip install opencv-python`.
*   **NumPy (`numpy`):** Used for creating the final array of frames and generating placeholder frames. Install via `pip install numpy`.
