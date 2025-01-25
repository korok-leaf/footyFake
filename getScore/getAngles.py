import pandas as pd
import cv2
import os

def getFrame(pose_path, ball_path, rightFoot, video_path):
    pp = pd.read_csv(pose_path)
    bp = pd.read_csv(ball_path)
    min_frame = 0
    min_diff = float('inf')

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(pp.shape[0], bp.shape[0])
    for i in range(min(pp.shape[0], bp.shape[0])):
        x_lm = pp.iloc[i]['landmark_28_x']*width
        y_lm = pp.iloc[i]['landmark_28_y']*height

        x1_ball = bp.iloc[i]['X1']
        y1_ball = bp.iloc[i]['Y1']
        x2_ball = bp.iloc[i]['X2']
        y2_ball = bp.iloc[i]['Y2'] 

        x_center = (x1_ball + x2_ball) / 2
        y_center = (y1_ball + y2_ball) / 2

        diff = (x_lm - x_center) ** 2 + (y_lm - y_center) ** 2
        print(diff)
        if diff < min_diff:
            min_diff = diff
            min_frame = i+1
    print(min_frame)
    print(getAngles(video_path, min_frame))

def getAngles(video_path, frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = cap.read()
    cap.release()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images', 'frame.jpg')
    cv2.imwrite(output_path, image)

    if not ret:
        print(f"Error: Could not read frame {frame}")
        return None


pose_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", 'pose_landmarks.csv')
soccer_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", 'soccer_ball_coordinates.csv')
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", 'videos', 'footy_video.mp4')

getFrame(pose_csv, soccer_csv, True, video_path)