

def getScore(
    knee_angle, 
    ankle_angle, 
    hip_angle, 
    plant_knee_angle, 
    torso_lean_angle, 
    shoulder_line_angle, 
    head_angle, 
    arm_angle
):
    
    optimal = {
        "knee_angle": 7.5,
        "ankle_angle": 25,
        "hip_angle": 90,
        "plant_knee_angle": 17.5,
        "torso_lean_angle": 15,
        "shoulder_line_angle": 2.5,
        "head_angle": 10,
        "arm_angle": 37.5
    }

    weights = {
        "knee_angle": 4,
        "ankle_angle": 3,
        "hip_angle": 3,
        "plant_knee_angle": 2,
        "torso_lean_angle": 2,
        "shoulder_line_angle": 1.5,
        "head_angle": 1.5,
        "arm_angle": 1
    }
    
    # Calculate penalty for each parameter
    penalties = {
        "knee_angle": abs(knee_angle - optimal["knee_angle"]) * weights["knee_angle"],
        "ankle_angle": abs(ankle_angle - optimal["ankle_angle"]) * weights["ankle_angle"],
        "hip_angle": abs(hip_angle - optimal["hip_angle"]) * weights["hip_angle"],
        "plant_knee_angle": abs(plant_knee_angle - optimal["plant_knee_angle"]) * weights["plant_knee_angle"],
        "torso_lean_angle": abs(torso_lean_angle - optimal["torso_lean_angle"]) * weights["torso_lean_angle"],
        "shoulder_line_angle": abs(shoulder_line_angle - optimal["shoulder_line_angle"]) * weights["shoulder_line_angle"],
        "head_angle": abs(head_angle - optimal["head_angle"]) * weights["head_angle"],
        "arm_angle": abs(arm_angle - optimal["arm_angle"]) * weights["arm_angle"],
    }

    score = 100 - sum(penalties.values())
    return max(score, 0)

print(getScore(7.5, 20, 100, 16, 15, 2.5, 20, 50))