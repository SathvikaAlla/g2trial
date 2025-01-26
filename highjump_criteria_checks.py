import cv2
from ultralytics import YOLO
import json
import numpy as np
import math

# from highjump_criteria_checks import get_bbox_center_xyxy, distance_2d, compute_speed, compute_angle_3pts


def draw_skeleton(frame, keypoints, color=(0, 255, 0)):
    """keypoints and skeleton lines on the frame."""
    POSE_CONNECTIONS = [
        (0, 1), (1, 3), (0, 2), (2, 4), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
        (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for kp in keypoints:
        x, y, conf = kp
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if (start_idx < len(keypoints) and end_idx < len(keypoints)
            and keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
            start_pt = tuple(map(int, keypoints[start_idx][:2]))
            end_pt   = tuple(map(int, keypoints[end_idx][:2]))
            cv2.line(frame, start_pt, end_pt, color, 2)

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
#   CRITERION 1 (Run-up)   #
#############################

def is_running_tall(keypoints, shoulder_margin=30):
    """
    checks if shoulders are at least 'shoulder_margin' pixels higher than hips.
      5 = left shoulder,  6 = right shoulder
      11 =left Hip,      12 = right hip
    """
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12

    shoulder_y = (keypoints[L_SHOULDER][1] + keypoints[R_SHOULDER][1]) / 2.0
    hip_y = (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2.0

    #if shoulders are significantly above hips, its "running tall"
    return (shoulder_y + shoulder_margin) < hip_y

def is_accelerating(speed_history, min_increase_count=3):
    """
    returns True if speed_history shows at least `min_increase_count` consecutive increases.
    eg:
      speed_history = [2.1, 2.5, 2.8, 3.0, 3.2]
      => consecutive increases = 4
    if min_increase_count=3 => True.
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

def evaluate_criterion1(keypoints, speed_history, criteria_state, frame_index):
    """
    checks if the athlete is:
      1) accelerating (speed increasing consecutively),
      2) running tall (shoulders above hips).
    if yes, sets 'criterion1_done' to True and increments 'score',
    and records the frame_index where it happened.
    """

    #1. Is accelerating?
    accelerating_ok = is_accelerating(speed_history)

    #2. Is running tall?
    tall_ok = is_running_tall(keypoints)

    if accelerating_ok and tall_ok:
        return True
    return False

#############################
#   CRITERION 2 (Leaning)  #
#############################

def check_lean_in_curve(keypoints, angle_thresh=150):
    """
    returns True if angle at the right shoulder (LeftShoulder->RightShoulder->RightHip)
    is below 'angle_thresh', indicating a leaning posture.
    """
    L_SHOULDER = 5
    R_SHOULDER = 6
    R_HIP = 12

    p_left_shoulder = (keypoints[L_SHOULDER][0], keypoints[L_SHOULDER][1])
    p_right_shoulder = (keypoints[R_SHOULDER][0], keypoints[R_SHOULDER][1])
    p_right_hip = (keypoints[R_HIP][0], keypoints[R_HIP][1])

    angle_deg = compute_angle_3pts(p_left_shoulder, p_right_shoulder, p_right_hip)
    return (angle_deg < angle_thresh)


#############
#Criterion 3#
#############
def check_knee_lift_at_takeoff(keypoints, angle_thresh=70):
    """
    criterion3 helper: check if leftHip->leftKnee->leftAnkle angle < angle_thresh.
    11L_Hip, 13L_Knee, 15L_Ankle
    """
    L_HIP, L_KNEE, L_ANKLE = 11, 13, 15
    p_hip = (keypoints[L_HIP][0], keypoints[L_HIP][1])
    p_knee = (keypoints[L_KNEE][0], keypoints[L_KNEE][1])
    p_ankle = (keypoints[L_ANKLE][0], keypoints[L_ANKLE][1])

    angle_deg = compute_angle_3pts(p_hip, p_knee, p_ankle)
    return (angle_deg < angle_thresh)


#################
#  Criterion 4  #
#################
def check_hollow_back(keypoints, angle_thresh=160):
    """
    criterion 4 helper: check if shoulders->hips->knees angle > angle_thresh (arched back).
    average left/right shoulders, left/right knees.
    """
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14

    #average shoulders
    shoulder_x = (keypoints[L_SHOULDER][0] + keypoints[R_SHOULDER][0]) / 2.0
    shoulder_y = (keypoints[L_SHOULDER][1] + keypoints[R_SHOULDER][1]) / 2.0
    #average hips
    hip_x = (keypoints[L_HIP][0] + keypoints[R_HIP][0]) / 2.0
    hip_y = (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2.0
    #average knees
    knee_x = (keypoints[L_KNEE][0] + keypoints[R_KNEE][0]) / 2.0
    knee_y = (keypoints[L_KNEE][1] + keypoints[R_KNEE][1]) / 2.0

    p_shoulder = (shoulder_x, shoulder_y)
    p_hip = (hip_x, hip_y)
    p_knee = (knee_x, knee_y)

    angle_deg = compute_angle_3pts(p_shoulder, p_hip, p_knee)
    return (angle_deg > angle_thresh)



#################
#  Criterion 5  #
#################

def check_l_shape_landing(keypoints, angle_range=(80, 100)):
    """
    criterion 5 helper: check if shoulders->hips->ankles forms ~90°.
    5 LShoulder, 6 RShoulder, 11 LHip, 12 RHip, 15 LAnkle, 16 RAnkle
    """
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_ANKLE, R_ANKLE = 15, 16

    #average shoulders
    shoulder_x = (keypoints[L_SHOULDER][0] + keypoints[R_SHOULDER][0]) / 2.0
    shoulder_y = (keypoints[L_SHOULDER][1] + keypoints[R_SHOULDER][1]) / 2.0
    #average hips
    hip_x = (keypoints[L_HIP][0] + keypoints[R_HIP][0]) / 2.0
    hip_y = (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2.0
    #average ankles
    ankle_x = (keypoints[L_ANKLE][0] + keypoints[R_ANKLE][0]) / 2.0
    ankle_y = (keypoints[L_ANKLE][1] + keypoints[R_ANKLE][1]) / 2.0

    p_shoulder = (shoulder_x, shoulder_y)
    p_hip = (hip_x, hip_y)
    p_ankle = (ankle_x, ankle_y)

    angle_deg = compute_angle_3pts(p_shoulder, p_hip, p_ankle)
    return (angle_deg >= angle_range[0] and angle_deg <= angle_range[1])

#############################
#            MAIN           #
#############################

def evaluate_high_jump(player_coords):

    scoring = {'high_runup':0, 'leaning_during_approach':0, 'knee_lift_at_takeoff':0, 'hollow_back_clearing_bar':0, 'l_shape_landing':0}
    DISPLACEMENT_THRESHOLD = 80.0  # start pixels to run-up
    runup_started = False
    initial_center = None
    center_previous = None
    speed_history = []

    evaluation_frames = {1:[], 2:[], 3:[], 4:[], 5:[]}


    for data in player_coords:
        frame = data['frame']
        kpts = data['keypoints']
        boxes = data['box']

        if boxes is not None:
            continue

        x1, y1, x2, y2 = map(int, boxes)

        if not runup_started:
            current_center = get_bbox_center_xyxy([x1, y1, x2, y2])
            if initial_center is None:
                initial_center = current_center # store it when bounding box appear or selected
            else:
                disp = distance_2d(current_center, initial_center) #how fat bounding box moved from original center
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

            if evaluate_criterion1(speed_history):
                scoring['high_runup'] = 1
                evaluation_frames[1].append(frame)
                
            if check_lean_in_curve(kpts):
                scoring['leaning_during_approach'] = 1
                evaluation_frames[2].append(frame)

            if check_knee_lift_at_takeoff(kpts):
                scoring['knee_lift_at_takeoff'] = 1
                evaluation_frames[3].append(frame)

            if check_hollow_back(kpts):
                scoring['hollow_back_clearing_bar'] = 1
                evaluation_frames[4].append(frame)

            if check_l_shape_landing(kpts):
                scoring['l_shape_landing'] = 1
                evaluation_frames[5].append(frame)
    
    return scoring, evaluation_frames

