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
        print(x_lm, y_lm, x1_ball, y1_ball, x2_ball, y2_ball)   

        if x1_ball <= x_lm <= x2_ball and y1_ball <= y_lm <= y2_ball:
            min_frame = i
    print(min_frame)


test = 'pose_landmarks.csv'
test2 = 'soccer_ball_coordinates.csv'
getAngles(test, test2, True)