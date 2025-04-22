# --- Import Necessary Libraries ---
import cv2  # OpenCV for video reading, writing, and image manipulation
import numpy as np # NumPy (imported but not directly used in this snippet)
import os  # Operating system library for path manipulation (basename, splitext, join, makedirs, rename, exists)
from ultralytics import YOLO  # YOLO object detection model from Ultralytics
import time  # Time library for tracking processing duration
from typing import Tuple, Set  # Type hinting for function signature

# --- Object Detection Function ---
def detection(path: str) -> Tuple[Set[str], str]: # Defines the main detection function
    """
    Detects and tracks objects in a video using YOLOv8 model, saving an annotated output video.
    
    Args:
        path (str): Path to the input video file. Supports common video formats (mp4, avi, etc.)
        
    Returns:
        Tuple[Set[str], str]: 
            - Set of unique detected object labels (e.g., {'Gun', 'Knife'})
            - Path to the output annotated video with detection boxes and tracking IDs
            
    Raises:
        FileNotFoundError: If input video doesn't exist.
        ValueError: If video cannot be opened or processed.
    """

    # --- Input Validation ---
    # Check if the provided video file path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}") # Raise error if not found

    # --- Model Initialization ---
    # Initialize the YOLOv8 model using the pre-trained weights file (.pt)
    # The comment indicates the classes this specific model is trained on.
    model = YOLO("D:/K_REPO/ComV/yolo/yolo/best.pt")
    # Get the mapping of class indices to class names (e.g., 0: 'Fire', 1: 'Gun', ...)
    class_names = model.names

    # --- Output Path Setup ---
    # Define the directory where results will be saved
    output_dir = r"D:\K_REPO\ComV\try\result"
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct filenames for temporary and final output videos
    input_video_name = os.path.basename(path) # Get filename from the input path
    base_name = os.path.splitext(input_video_name)[0] # Get filename without extension
    # Temporary filename used during processing
    temp_output_name = f"{base_name}_output_temp.mp4"
    # Full path for the temporary output video
    temp_output_path = os.path.join(output_dir, temp_output_name)

    # --- Video Processing Setup ---
    # Open the input video file for reading frames
    cap = cv2.VideoCapture(path)
    # Check if the video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {path}") # Raise error if opening failed

    # Define the desired output frame dimensions (resizing ensures consistency)
    frame_width, frame_height = 640, 640
    # Initialize the VideoWriter object to save the annotated video
    out = cv2.VideoWriter(
        temp_output_path, # Path to save the temporary output video
        cv2.VideoWriter_fourcc(*'mp4v'),  # Codec for MP4 video format
        30.0,  # Desired frames per second for the output video
        (frame_width, frame_height) # Frame size (width, height)
    )

    # --- Main Processing Loop ---
    # Initialize a list to store the names of all detected objects across frames
    crimes = []
    # Record the start time for performance measurement
    start = time.time()
    print(f"[INFO] Processing started at {start:.2f} seconds")

    # Loop through each frame of the input video
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        # If 'ret' is False, it means the end of the video is reached
        if not ret:
            break # Exit the loop

        # --- Frame Processing ---
        # Resize the current frame to the standard processing dimensions
        frame = cv2.resize(frame, (frame_width, frame_height))
        # Run the YOLOv8 model's tracking function on the resized frame
        results = model.track(
            source=frame, # The input frame
            conf=0.7,  # Confidence threshold: only detections with score >= 0.7 are considered
            persist=True  # Enable object tracking across consecutive frames
        )

        # --- Annotation and Output ---
        # Use the model's built-in plot() function to draw bounding boxes and tracking IDs on the frame
        annotated_frame = results[0].plot()
        
        # --- Record Detections ---
        # Iterate through the detected bounding boxes in the current frame's results
        for box in results[0].boxes:
            # Get the class index of the detected object
            cls = int(box.cls)
            # Append the corresponding class name to the 'crimes' list
            crimes.append(class_names[cls])

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

    # --- Cleanup and Finalization ---
    # Record the end time
    end = time.time()
    print(f"[INFO] Processing finished at {end:.2f} seconds")
    # Calculate and print the total processing time
    print(f"[INFO] Total execution time: {end - start:.2f} seconds")
    
    # Release the video capture and writer objects to free resources
    cap.release()
    out.release()

    # --- Final Output Naming ---
    # Create a set of unique detected object names
    unique_crimes = set(crimes)
    # Create a string representation of the unique crimes, sorted alphabetically,
    # joined by underscores, spaces replaced, and truncated to 50 chars for filename sanity.
    crimes_str = "_".join(sorted(unique_crimes)).replace(" ", "_")[:50]
    # Construct the final output filename including the detected object names
    final_output_name = f"{base_name}_{crimes_str}_output.mp4"
    # Construct the full path for the final output video
    final_output_path = os.path.join(output_dir, final_output_name)

    # Rename the temporary output file to the final filename
    # This avoids issues if the script crashes mid-processing
    try:
        os.rename(temp_output_path, final_output_path)
    except FileExistsError:
        # Handle case where the final file might already exist (e.g., re-running)
        os.remove(final_output_path) # Remove existing file
        os.rename(temp_output_path, final_output_path) # Try renaming again


    # Print the set of unique detected objects and the final output path
    print(f"[INFO] Detected crimes: {unique_crimes}")
    print(f"[INFO] Annotated video saved at: {final_output_path}")

    # Return the set of unique detected object labels and the path to the final annotated video
    return unique_crimes, final_output_path


# --- Example Usage (Commented Out) ---
# # Entry point for running the script directly (currently commented out)
# path0 = input("Enter the local path to the video file to detect objects: ") # Prompt user for video path
# path = path0.strip('"')  # Remove potential surrounding quotes
# print(f"[INFO] Loading video: {path}") # Print informational message
# detection(path) # Call the detection function
