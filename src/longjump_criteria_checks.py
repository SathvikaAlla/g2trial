from ultralytics import YOLO
import json
import numpy as np
import math

#############################
#    HELPER FUNCTIONS      #
#############################


def get_bbox_center_xyxy(box):
    """returns bounding box center (cx, cy)."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)

def distance_2d(p1, p2):
    """euclidean distance between two points p1=(x1, y1), p2=(x2, y2)."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def compute_speed(center_curr, center_prev):
    """approx. speed in pixels/frame between consecutive bounding box centers."""
    if center_prev is None:
        return 0.0
    dx = center_curr[0] - center_prev[0]
    dy = center_curr[1] - center_prev[1]
    return math.hypot(dx, dy)

def compute_angle_3pts(p1, p2, p3):
    """
    computes angle (in degrees) at p2 for points p1->p2->p3.
    returns angle in [0..180].
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]

    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0.0

    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos_angle = dot / (mag1*mag2)
    cos_angle = max(-1, min(1, cos_angle))  # clamp
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg

#############################
#      CRITERION LOGIC     #
#############################

#################
#  criterion 1  #
#################

# 1. Run-up is accelerating, no stutter steps
def is_accelerating(speed_history, min_increase_count=3):
    """
    if speed is consistently rising for min_increase_count times -> accelerating.
    """
    if len(speed_history) < min_increase_count + 1:
        return False
    consecutive_increases = 0
    for i in range(1, len(speed_history)):
        if speed_history[i] > speed_history[i - 1]:
            consecutive_increases += 1
        else:
            consecutive_increases = 0
        if consecutive_increases >= min_increase_count:
            return True
    return False


#################
#  criterion 2  #
#################


#board_x_min, board_x_max, board_y_min, board_y_max
BOARD_REGION = (195, 230, 350, 400) #will adjust it base on the video quility

def foot_on_board(kpts, board_region):
    """
    checks if the right foot/ankle keypoint is within the bounding box board_region.
    board_region = (xmin, xmax, ymin, ymax)
    kpts: shape (17,3), x,y,conf for each keypoint
    returns True if the foot is inside that bounding region.
    """
    RIGHT_ANKLE = 16
    foot_x = kpts[RIGHT_ANKLE][0]
    foot_y = kpts[RIGHT_ANKLE][1]

    xmin, xmax, ymin, ymax = board_region
    if (xmin <= foot_x <= xmax) and (ymin <= foot_y <= ymax):
        return True
    return False


# 2.foot on the board, not looking at board
def check_not_looking_down(kpts):
    """
    if nose is above eyes, not looking down.
    (Placeholder approach)
    """
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    if kpts[NOSE][1] < kpts[L_EYE][1] and kpts[NOSE][1] < kpts[R_EYE][1]:
        return True
    return False

def evaluate_criterion2(kpts):
    """
    Niet kijken naar de afstootbalk en afstootvoet is op de witte plank
    - foot is on board region
    - nose is not lower than eyes -> not looking down
    """


    foot_ok = foot_on_board(kpts, BOARD_REGION)   #real bounding region check
    head_ok = check_not_looking_down(kpts)

    if foot_ok and head_ok:
        return True
    return False


#################
#  criterion 3  #
#################
# 3.foot is flat on ground, center of mass above foot
def check_foot_flat_and_com_over_foot(kpts):
    """
    angle check for the foot, if the ankle/knee/hip align vertically.
    """
    R_ANKLE = 16
    R_KNEE  = 14
    R_HIP   = 12
    # angle at the ankle
    p_ankle = (kpts[R_ANKLE][0], kpts[R_ANKLE][1])
    p_knee  = (kpts[R_KNEE][0],  kpts[R_KNEE][1])
    p_hip   = (kpts[R_HIP][0],   kpts[R_HIP][1])

    angle_deg = compute_angle_3pts(p_ankle, p_knee, p_hip)
    #if angle is fairly straight (eg:170–180 deg), foot is flat body is stacked.
    foot_flat_ok = (angle_deg > 165)

    #check if 'center of mass' is above foot by compare x coords
    return foot_flat_ok

#################
#  criterion 4  #
#################
# 4. takeoff leg not pulled in too early -> "riddersstand"
def check_takeoff_leg_not_pulled(kpts):
    """
    "leg not pulled in too early" can be knee angle not too bent in mid-air. check left knee angle if it's a left takeoff.
    """
    L_HIP   = 11
    L_KNEE  = 13
    L_ANKLE = 15
    p_hip   = (kpts[L_HIP][0],   kpts[L_HIP][1])
    p_knee  = (kpts[L_KNEE][0],  kpts[L_KNEE][1])
    p_ankle = (kpts[L_ANKLE][0], kpts[L_ANKLE][1])

    angle_deg = compute_angle_3pts(p_hip, p_knee, p_ankle)
    #if angle is > 120, maybe it's may not too bent
    return (angle_deg > 120)


#################
#  criterion 5  #
#################
# 5. landing happens with sliding technique
def check_sliding_landing(kpts):
    """
    check if feet and butt are near same horizontal line
    as they land. then check angle, shoulders->hips->feet.
    """
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP           = 11, 12
    L_ANKLE, R_ANKLE       = 15, 16

    #average shoulders
    sh_x = (kpts[L_SHOULDER][0] + kpts[R_SHOULDER][0]) / 2.0
    sh_y = (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2.0
    #average hips
    hip_x = (kpts[L_HIP][0] + kpts[R_HIP][0]) / 2.0
    hip_y = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2.0
    #average ankles
    an_x = (kpts[L_ANKLE][0] + kpts[R_ANKLE][0]) / 2.0
    an_y = (kpts[L_ANKLE][1] + kpts[R_ANKLE][1]) / 2.0

    p_shoulder = (sh_x, sh_y)
    p_hip      = (hip_x, hip_y)
    p_ankle    = (an_x, an_y)

    angle_deg = compute_angle_3pts(p_shoulder, p_hip, p_ankle)
    #if angle is  90 deg then it might be a sliding posture
    if 80 <= angle_deg <= 100:
        return True
    return False



#############################
#         MAIN CODE        #
#############################
def evaluate_long_jump(player_coords):
    scoring = {'runup_started': 0, 'accelerating_runup': 0, 'foot_on_board': 0,
               'foot_flat_COM_above': 0, 'takeoff_leg_extension': 0, 'sliding_landing': 0}

    # Initialize lists to track speed and keypoints over time
    speed_history = []
    evaluation_frames = {1: [], 2: [], 3: [], 4: [], 5: []}

    runup_started = False
    initial_center = None
    center_previous = None
    DISPLACEMENT_THRESHOLD = 80.0  # Displacement threshold to detect run-up start

    for data in player_coords:
        frame = data['frame']
        kpts = data['keypoints']
        boxes = data['box']

        if not boxes or not kpts:
            continue

        # Assuming only one athlete per frame; extract box and keypoints
        x1, y1, x2, y2 = map(int, boxes)  # Extract bounding box (assuming only one box per frame)

        # Calculate bounding box center and check for run-up
        
        if not runup_started:
            current_center = get_bbox_center_xyxy([x1, y1, x2, y2])
            if initial_center is None:
                initial_center = current_center
            else:
                disp = distance_2d(current_center, initial_center)
                if disp > DISPLACEMENT_THRESHOLD:
                    runup_started = True
                    center_previous = current_center
                    speed_history.clear()
        else:
            current_center = get_bbox_center_xyxy([x1, y1, x2, y2])
            speed = compute_speed(current_center, center_previous)
            center_previous = current_center
            if speed > 0:
                speed_history.append(speed)

            # Criterion 1: Accelerating run-up
            if is_accelerating(speed_history):
                scoring['accelerating_runup'] = 1
                evaluation_frames[2].append(frame)

            # Criterion 2: Foot on board (simplified assumption of foot position)
            if evaluate_criterion2(kpts):
                scoring['foot_on_board'] = 1
                evaluation_frames[3].append(frame)

            # Criterion 3: Foot flat, COM above foot (simplified check)
            if check_foot_flat_and_com_over_foot(kpts):
                scoring['foot_flat_COM_above'] = 1
                evaluation_frames[4].append(frame)

            # Criterion 4: Takeoff leg full extension (simplified check)
            if check_takeoff_leg_not_pulled(kpts):
                scoring['takeoff_leg_extension'] = 1
                evaluation_frames[5].append(frame)

            # Criterion 5: Sliding landing (simplified check)
            if check_sliding_landing(kpts):
                scoring['sliding_landing'] = 1
                evaluation_frames[5].append(frame)

    return scoring, evaluation_frames
