import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
import numpy as np


def make_pose_model(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)
    
    return options
# Initialize MediaPipe's Image module
mp_image_module = mp.Image



BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

def process_video(video_path):
    """
    Processes the input video, converts each frame to a MediaPipe Image object,
    and prints the timestamp for each frame.
    
    Args:
        video_path (str): Path to the input video file.
    """
    
    # #1. Use OpenCV’s VideoCapture to load the input video.
    cap = cv2.VideoCapture(video_path)
    
    # #2. Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Frames per second (FPS): {fps}")
    
    # Calculate the duration of each frame
    frame_duration = 1 / fps
    print(f"Duration of each frame: {frame_duration:.4f} seconds")
    
    frame_count = 0  # To keep track of frame numbers
    
    # #3. Loop through each frame in the video using VideoCapture.read()
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or cannot fetch the frame.")
            break
        
        # Calculate the timestamp for the current frame
        timestamp = frame_count * frame_duration
        print(f"Processing Frame {frame_count}: Timestamp {timestamp:.2f} seconds")
        
        # #4. Convert the frame received from OpenCV to a MediaPipe’s Image object.
        # OpenCV reads images in BGR format; convert it to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe Image object
        mp_image = mp_image_module.Image(
            image_format=mp_image_module.ImageFormat.SRGB, 
            data=frame_rgb
        )
        
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        # Example: You can now pass `mp_image` to MediaPipe processing functions
        # For demonstration, we'll just print the type
        print(f"Converted to MediaPipe Image: {type(mp_image)}")
        
        # Increment frame count
        frame_count += 1
        
        # Optional: Display the frame (press 'q' to quit)
        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback interrupted by user.")
            break
    
    # Release the VideoCapture object and close display window
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

if __name__ == "__main__":
    # Replace 'input_video.mp4' with your video file path
    video_path = "videos/footy_video.mp4"
    model_path = "da_models/pose_landmarker_heavy.task"
    options = make_pose_model(model_path)
    process_video(video_path)

    with PoseLandmarker.create_from_options(options) as landmarker:
        process_video(video_path, landmarker)
    