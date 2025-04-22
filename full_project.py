# --- Import Necessary Libraries ---
import cv2  # OpenCV for potential video processing (though not directly used here)
import numpy as np # NumPy for numerical operations (though not directly used here)
import os # Operating system library (though not directly used here)
from ultralytics import YOLO # YOLO object detection model (imported but seems unused directly, likely used within 'detection')
import time # Time library (though not directly used here)
import tensorflow as tf # TensorFlow for deep learning (imported but seems unused directly, likely used within 'fight_detec')
from frame_slicer import extract_video_frames # Frame extraction utility (imported but seems unused directly, likely used within 'fight_detec')
import matplotlib.pyplot as plt # Matplotlib for plotting (imported but seems unused directly, likely used within 'fight_detec')

# --- Import Custom Modules ---
# Import the fight detection function from the 'Fight_detec_func.py' file
from Fight_detec_func import fight_detec
# Import the object detection function from the 'objec_detect_yolo.py' file
from objec_detect_yolo import detection


# --- Main Execution Block ---
# This code runs when the script is executed directly.

# Prompt the user to enter the path to the video file
path0 = input("Enter the local path : ")
# Remove any surrounding quotation marks from the input path (useful for paths copied from Windows Explorer)
path = path0.strip('"')
# Print an informational message indicating which video is being processed
print(f"[INFO] Loading video: {path}")

# Call the fight detection function with the provided video path
# This will analyze the video for fight activity and print/save results (depending on the function's implementation)
fight_detec(path)

# Call the object detection function with the provided video path
# This will analyze the video to detect objects (e.g., persons, weapons) and print/save results
detection(path)
