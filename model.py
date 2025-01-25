import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os
from mediapipe.python._framework_bindings import timestamp

def make_pose_model(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)
    
    return options

def initialize_landmarker(options):
    """
    Initializes and returns the PoseLandmarker based on the provided options.

    Args:
        options (PoseLandmarkerOptions): Configuration options for PoseLandmarker.

    Returns:
        PoseLandmarker: An instance of PoseLandmarker.
    """
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    landmarker = PoseLandmarker.create_from_options(options)
    return landmarker


def initialize_csv(csv_path, num_landmarks):
    """
    Initializes a CSV file with appropriate headers to store pose landmarks.

    Args:
        csv_path (str): Path to the output CSV file.
        num_landmarks (int): Number of pose landmarks per frame.
    
    Returns:
        pandas.DataFrame: An empty DataFrame with the correct columns.
    """
    # Define column names: frame_number, timestamp, then landmark coordinates
    columns = ['frame_number', 'timestamp']
    for i in range(num_landmarks):
        columns.extend([
            f'landmark_{i}_x',
            f'landmark_{i}_y',
            f'landmark_{i}_z',
            f'landmark_{i}_visibility',
            f'landmark_{i}_presence'
        ])
    
    # Create an empty DataFrame with the defined columns
    df = pd.DataFrame(columns=columns)
    
    # Save headers to CSV
    df.to_csv(csv_path, index=False)
    
    return df

def append_to_csv(csv_path, frame_data):
    """
    Appends a single frame's pose data to the CSV file.

    Args:
        csv_path (str): Path to the output CSV file.
        frame_data (dict): Dictionary containing frame data to append.
    """
    df = pd.DataFrame([frame_data])
    df.to_csv(csv_path, mode='a', header=False, index=False)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def process_video(video_path, landmarker, csv_path):
    """
    Processes the input video, extracts pose landmarks for each frame,
    and saves them into a CSV file.

    Args:
        video_path (str): Path to the input video file.
        landmarker (PoseLandmarker): Initialized PoseLandmarker instance.
        csv_path (str): Path to the output CSV file.
    """
    
    # Initialize OpenCV's VideoCapture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    # Retrieve frame rate (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS value is 0. Cannot process video.")
        cap.release()
        return
    print(f"Frames per second (FPS): {fps}")
    
    # Calculate the duration of each frame in seconds
    frame_duration = 1 / fps
    print(f"Duration of each frame: {frame_duration:.4f} seconds")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"Total frames in video: {total_frames}")
    
    frame_count = 0  # To keep track of frame numbers
    
    # Initialize CSV file
    num_landmarks = 33  # MediaPipe Pose has 33 landmarks
    df = initialize_csv(csv_path, num_landmarks)
    
    # Loop through each frame in the video
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or cannot fetch the frame.")
            break
        
        # Calculate the timestamp for the current frame in milliseconds
        current_timestamp = frame_count * frame_duration * 1000  # Convert to ms
        
        print(f"Processing Frame {frame_count}: Timestamp {current_timestamp:.2f} ms")
        
        # Convert frame from BGR (OpenCV) to RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, 
            data=frame_rgb
        )
        
        # Perform pose detection
        pose_landmarker_result = landmarker.detect_for_video(mp_image, int(current_timestamp))
        
        if pose_landmarker_result.pose_landmarks:
            print(pose_landmarker_result.pose_landmarks)
            landmarks = pose_landmarker_result.pose_landmarks[0]
            frame_data = {
                'frame_number': frame_count,
                'timestamp': current_timestamp
            }
            for i, landmark in enumerate(landmarks):
                frame_data[f'landmark_{i}_x'] = landmark.x
                frame_data[f'landmark_{i}_y'] = landmark.y
                frame_data[f'landmark_{i}_z'] = landmark.z
                frame_data[f'landmark_{i}_visibility'] = landmark.visibility
                frame_data[f'landmark_{i}_presence'] = landmark.presence
        else:
            # If no landmarks detected, fill with NaNs
            frame_data = {
                'frame_number': frame_count,
                'timestamp': current_timestamp
            }
            for i in range(num_landmarks):
                frame_data[f'landmark_{i}_x'] = np.nan
                frame_data[f'landmark_{i}_y'] = np.nan
                frame_data[f'landmark_{i}_z'] = np.nan
                frame_data[f'landmark_{i}_visibility'] = np.nan
                frame_data[f'landmark_{i}_presence'] = np.nan
        
        # Append the frame data to the CSV
        append_to_csv(csv_path, frame_data)
        
        # Increment frame count
        frame_count += 1
        
        # Optional: Display the frame with pose landmarks
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        width = annotated_image.shape[1]
        height = annotated_image.shape[0]
        resized_frame = cv2.resize(annotated_image, (width // 2, height // 2))
        cv2.imshow('Annotated Video Frame', cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback interrupted by user.")
            break
    
    # Release the VideoCapture object and close display window
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

def main():
    # Define paths
    video_path = "videos/footy_video.mp4"  # Replace with your video path
    model_path = "da_models/pose_landmarker_heavy.task"  # Replace with your model path
    csv_output_path = "pose_landmarks.csv"  # Output CSV file
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Initialize MediaPipe's Image module
    mp_image_module = mp.Image
    
    # Create PoseLandmarker options
    options = make_pose_model(model_path)
    
    # Initialize PoseLandmarker
    landmarker = initialize_landmarker(options)
    
    try:
        # Process the video and save pose landmarks to CSV
        process_video(video_path, landmarker, csv_output_path)
    finally:
        # Ensure the landmarker is closed properly
        landmarker.close()
        print(f"Pose landmarks saved to {csv_output_path}")

if __name__ == "__main__":
    main()