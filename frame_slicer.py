import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy library for numerical operations
import random # Random library (though not used in the current function)

def extract_video_frames(video_path, n_frames=30, frame_size=(96, 96)): # Defines the function to extract frames
    """
    Simplified robust frame extractor for short videos (2-10 sec)
    - Automatically handles varying video lengths
    - Ensures consistent output shape
    - Optimized for MP4/MPEG
    """
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}") # Print error if opening failed
        return None # Return None if video cannot be opened
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Basic validation to check if the video has frames and a valid FPS
    if total_frames < 1 or fps < 1:
        print(f"Error: Invalid video (frames:{total_frames}, fps:{fps})") # Print error for invalid video
        cap.release() # Release the video capture object
        return None # Return None for invalid video
    
    # Calculate the approximate length of the video in seconds
    video_length = total_frames / fps
    # Calculate the step size to skip frames, ensuring at least 1
    frame_step = max(1, int(total_frames / n_frames))
    
    frames = [] # Initialize an empty list to store the extracted frames
    last_good_frame = None # Initialize a variable to store the last successfully read frame
    
    # Loop to extract the desired number of frames (n_frames)
    for i in range(n_frames):
        # Calculate the frame position to read, ensuring it's within bounds
        pos = min(int(i * (total_frames / n_frames)), total_frames - 1)
        # Set the video capture position to the calculated frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        
        # Read the frame at the current position
        ret, frame = cap.read()
        
        # Handle cases where reading the frame fails (ret is False or frame is None)
        if not ret or frame is None:
            # If a previous good frame exists, use a copy of it
            if last_good_frame is not None:
                frame = last_good_frame.copy()
            # Otherwise, generate a placeholder frame (light gray)
            else:
                # Create a NumPy array filled with 0.8 (light gray), matching the desired frame size and 3 color channels
                frame = np.full((*frame_size[::-1], 3), 0.8, dtype=np.float32)
        # If the frame was read successfully
        else:
            # Resize the frame to the specified frame_size
            frame = cv2.resize(frame, frame_size)
            # Convert the frame from BGR (OpenCV default) to RGB color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame data type to float32 and normalize pixel values to the range [0.0, 1.0]
            frame = frame.astype(np.float32) / 255.0
            # Update the last good frame
            last_good_frame = frame
        
        # Append the processed (or placeholder) frame to the list
        frames.append(frame)
    
    # Release the video capture object to free up resources
    cap.release()
    # Convert the list of frames into a NumPy array and return it
    return np.array(frames)
