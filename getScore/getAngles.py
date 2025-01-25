import pandas as pd

def getAngles(pose_path, ball_path, foot):
    pp = pd.read_csv(pose_path)
    bp = pd.read_csv(ball_path)

    min_frame = 0
    min_diff = float('inf')
    for i in range(min(pp.shape[0], bp.shape[0])):
        x_lm = pp.iloc[i]['landmark_28_x']
        y_lm = pp.iloc[i]['landmark_28_y']

        x1_ball = bp.iloc[i]['X1']
        y1_ball = bp.iloc[i]['Y1']
        x2_ball = bp.iloc[i]['X2']
        y2_ball = bp.iloc[i]['Y2'] 

        x_center = (x1_ball + x2_ball) / 2
        y_center = (y1_ball + y2_ball) / 2

        diff = (x_lm - x_center) ** 2 + (y_lm - y_center) ** 2
        if diff < min_diff:
            min_diff = diff
            min_frame = i
    print(min_frame)


test = 'pose_landmarks.csv'
test2 = 'soccer_ball_coordinates.csv'
getAngles(test, test2, True)