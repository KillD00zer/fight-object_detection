import tensorflow as tf  # TensorFlow library for deep learning
from frame_slicer import extract_video_frames  # Import the frame extraction function from another file
import cv2  # OpenCV library for computer vision tasks
import os  # Operating system library for path manipulation
import numpy as np  # NumPy library for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting visualizations

# --- Configuration ---
MODEL_PATH = r"D:\K_REPO\ComV\AI_made\trainnig_output\final_model_2.h5"  # Path to the pre-trained Keras model file
N_FRAMES = 30  # Number of frames to extract from each video
IMG_SIZE = (96, 96)  # Target size to resize each frame to
RESULT_PATH = r"D:\K_REPO\ComV\try\result"  # Directory to save debug outputs and visualizations

# --- Main Detection Function ---
def fight_detec(video_path: str, debug: bool = True):
    """
    Detects fight activity in a given video file.

    Args:
        video_path (str): The path to the video file.
        debug (bool, optional): If True, prints debug information and saves visualizations. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - str: The prediction result ("FIGHT" or "NORMAL" with confidence) or an error message.
            - float or None: The raw prediction score from the model, or None if an error occurred.
    """
    
    # --- Helper Class for Detection Logic ---
    class FightDetector:
        """Encapsulates the model loading, frame extraction, and prediction logic."""
        def __init__(self):
            """Initializes the FightDetector by loading the model."""
            self.model = self._load_model() # Load the model upon instantiation
        
        def _load_model(self):
            """Loads the pre-trained Keras model."""
            try:
                # Load the model from the specified path, disable compilation (often faster for inference)
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                if debug:
                    # Print success message and model input shape if debug mode is on
                    print("\nModel loaded successfully. Input shape:", model.input_shape)
                return model # Return the loaded model
            except Exception as e:
                # Print error message if loading fails
                print(f"Model loading failed: {e}")
                return None # Return None if loading failed
        
        def _extract_frames(self, video_path):
            """Extracts and preprocesses frames from the video."""
            # Use the imported function to extract frames
            frames = extract_video_frames(video_path, N_FRAMES, IMG_SIZE)
            if frames is None:
                # Return None if frame extraction failed
                return None
            
            # Debugging steps if debug mode is enabled
            if debug:
                # Count how many frames are completely black (all pixel values are 0)
                blank_frames = np.all(frames == 0, axis=(1, 2, 3)).sum()
                if blank_frames > 0:
                    # Print a warning if blank frames are detected
                    print(f"Warning: {blank_frames} blank frames detected")
                # Save a sample frame for visual inspection
                sample_frame = (frames[0] * 255).astype(np.uint8) # Convert first frame back to 0-255 range
                # Save the frame as a JPG image in the result path
                cv2.imwrite(os.path.join(RESULT_PATH, 'debug_frame.jpg'), 
                            cv2.cvtColor(sample_frame, cv2.COLOR_RGB2BGR)) # Convert RGB back to BGR for OpenCV saving
            
            return frames # Return the extracted and processed frames
        
        def predict(self, video_path):
            """Performs fight detection prediction on the video."""
            # Check if the video file exists
            if not os.path.exists(video_path):
                return "Error: Video not found", None # Return error if file doesn't exist
            
            try:
                # Extract frames from the video
                frames = self._extract_frames(video_path)
                if frames is None:
                    # Return error if frame extraction failed
                    return "Error: Frame extraction failed", None
                
                # Validate the number of extracted frames
                if frames.shape[0] != N_FRAMES:
                    # Return error if the number of frames doesn't match the expected count
                    return f"Error: Expected {N_FRAMES} frames, got {frames.shape[0]}", None
                
                # Check if all extracted frames are blank (potentially problematic video)
                if np.all(frames == 0):
                    return "Error: All frames are blank", None # Return error if all frames are blank
                
                # Perform prediction using the loaded model
                # Add a batch dimension (np.newaxis) as the model expects a batch of sequences
                # verbose=0 suppresses prediction progress output
                prediction = self.model.predict(frames[np.newaxis, ...], verbose=0)[0][0] # Get the single prediction score
                
                # Determine the result based on a threshold (0.61)
                result = "FIGHT" if prediction >= 0.61 else "NORMAL"
                # Calculate a confidence score (heuristic, scales the distance from threshold)
                confidence = min(max(abs(prediction - 0.61) * 150 + 50, 0), 100) # Clamp between 0 and 100
                
                # If debug mode is on, generate and save a visualization
                if debug:
                    self._debug_visualization(frames, prediction, result, video_path)
                
                # Return the formatted result string and the raw prediction score
                return f"{result} ({confidence:.1f}% confidence)", prediction
            
            # Catch any exceptions during the prediction process
            except Exception as e:
                return f"Prediction error: {str(e)}", None # Return a generic prediction error message

        def _debug_visualization(self, frames, score, result, video_path):
            """Generates and saves a visualization of sample frames and the prediction."""
            print(f"\nPrediction Score: {score:.4f}") # Print the raw prediction score
            print(f"Decision: {result}") # Print the final decision
            plt.figure(figsize=(15, 5)) # Create a matplotlib figure
            # Display up to the first 10 frames
            for i in range(min(10, len(frames))):
                plt.subplot(2, 5, i+1) # Create subplots for each frame
                plt.imshow(frames[i]) # Display the frame (already in RGB format)
                plt.title(f"Frame {i}\nMean: {frames[i].mean():.2f}") # Show frame index and mean pixel value
                plt.axis('off') # Hide axes
            plt.suptitle(f"Prediction: {result} (Score: {score:.4f})") # Set the main title for the figure
            plt.tight_layout() # Adjust layout to prevent overlapping titles

            # Save the visualization figure
            base_name = os.path.splitext(os.path.basename(video_path))[0] # Get the video filename without extension
            save_path = os.path.join(RESULT_PATH, f"{base_name}_prediction_result.png") # Construct the save path
            plt.savefig(save_path) # Save the figure
            plt.close() # Close the plot to free memory
            print(f"Visualization saved to: {save_path}") # Print the path where the visualization was saved

    # --- Execution ---
    # Create an instance of the FightDetector class
    detector = FightDetector()
    # Check if the model loaded successfully
    if detector.model is None:
        return "Error: Model loading failed", None # Return error if model loading failed during init
    # Call the predict method on the detector instance and return its result
    return detector.predict(video_path)

# --- Example Usage (Commented Out) ---
# # Entry point for running the script directly (currently commented out)
# path0 = input("Enter the local path to the video file to detect fight: ") # Prompt user for video path
# path = path0.strip('"')  # Remove potential surrounding quotes from path (e.g., when copying from Windows Explorer)
# print(f"[INFO] Loading video: {path}") # Print informational message
# fight_detec(path) # Call the main detection function
