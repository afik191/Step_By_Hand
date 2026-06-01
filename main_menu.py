import cv2
import sys
import math
import time
import subprocess
import mediapipe as mp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPT_RPS = BASE_DIR / "GameMode" / "game_Rock_Paper_Scissors.py"
SCRIPT_EVEN_ODD = BASE_DIR / "GameMode" / "game_Even_Odd.py"

SCRIPT_COUNTING = BASE_DIR / "LearningMode" / "countingMode.py"
SCRIPT_MATH = BASE_DIR / "LearningMode" / "game_Plus_Minus.py"
SCRIPT_GREATER_SMALLER = BASE_DIR / "LearningMode" / "game_Big_Small.py"

MAIN_MENU = "MAIN_MENU"
GAME_MENU = "GAME_MENU"
EDUCATION_MENU = "EDUCATION_MENU"

menu_state = MAIN_MENU
last_selection_time = 0
SELECTION_COOLDOWN = 1.2
RETURN_COOLDOWN = 2.0

cap = None
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers_from_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return -1

    fingers = []
    if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6:
        fingers.append(1)
    else:
        fingers.append(0)

    tips_idx, mips_idx = [8, 12, 16, 20], [6, 10, 14, 18]
    for tip, mip in zip(tips_idx, mips_idx):
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip])
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip])
        if mip_dist > 0 and tip_dist / mip_dist > 1.15:
            fingers.append(1)
        else:
            fingers.append(0)
    return max(0, min(5, sum(fingers)))

def is_back_gesture(hand_landmarks):
    """
    מחוות חזרה: סימן OK (👌)
    קצה האגודל קרוב לקצה האצבע המורה, ושאר האצבעות פתוחות.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False

    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    
    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0: return False
        return (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) > 1.10

    return thumb_index_dist < hand_scale * 0.35 and is_finger_open(12, 10) and is_finger_open(16, 14) and is_finger_open(20, 18)

def open_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened(): sys.exit(1)
    return camera

def release_camera():
    global cap
    if cap is not None and cap.isOpened(): cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.5)

def run_mode(script_path):
    global cap, last_selection_time
    if not script_path.exists(): return
    release_camera()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(script_path.parent))
    time.sleep(0.8)
    cap = open_camera()
    last_selection_time = time.time() + RETURN_COOLDOWN

def draw_main_menu(frame):
    cv2.putText(frame, "MAIN MENU", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(frame, "Show 1 finger: GAME MODE", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "Show 2 fingers: EDUCATION MODE", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

def draw_game_menu(frame):
    cv2.putText(frame, "GAME MODE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(frame, "Show 1 finger: Rock Paper Scissors", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "Show 2 fingers: Even / Odd", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "👌 (OK Sign): return to Main Menu", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

def draw_education_menu(frame):
    cv2.putText(frame, "EDUCATION MODE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(frame, "Show 1 finger: Counting / Imitation", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "Show 2 fingers: Addition / Subtraction", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "Show 3 fingers: Greater / Smaller", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "👌 (OK Sign): return to Main Menu", (40, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

cap = open_camera()

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while True:
            if cap is None or not cap.isOpened(): cap = open_camera()
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            current_fingers = -1
            back_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_fingers = count_fingers_from_landmarks(hand_landmarks)
                    back_detected = is_back_gesture(hand_landmarks)

            current_time = time.time()

            if menu_state == MAIN_MENU: draw_main_menu(frame)
            elif menu_state == GAME_MENU: draw_game_menu(frame)
            elif menu_state == EDUCATION_MENU: draw_education_menu(frame)

            if current_fingers != -1:
                cv2.putText(frame, f"Detected fingers: {current_fingers}", (40, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if back_detected:
                cv2.putText(frame, "Back gesture (OK) detected", (40, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            if current_time - last_selection_time > SELECTION_COOLDOWN:
                if menu_state == MAIN_MENU:
                    if current_fingers == 1:
                        menu_state = GAME_MENU
                        last_selection_time = current_time
                    elif current_fingers == 2:
                        menu_state = EDUCATION_MENU
                        last_selection_time = current_time
                elif menu_state == GAME_MENU:
                    if back_detected:
                        menu_state = MAIN_MENU
                        last_selection_time = current_time
                    elif current_fingers == 1:
                        last_selection_time = current_time
                        run_mode(SCRIPT_RPS)
                    elif current_fingers == 2:
                        last_selection_time = current_time
                        run_mode(SCRIPT_EVEN_ODD)
                elif menu_state == EDUCATION_MENU:
                    if back_detected:
                        menu_state = MAIN_MENU
                        last_selection_time = current_time
                    elif current_fingers == 1:
                        last_selection_time = current_time
                        run_mode(SCRIPT_COUNTING)
                    elif current_fingers == 2:
                        last_selection_time = current_time
                        run_mode(SCRIPT_MATH)
                    elif current_fingers == 3:
                        last_selection_time = current_time
                        run_mode(SCRIPT_GREATER_SMALLER)

            cv2.imshow("Robotic Hand - Main Interface", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
finally:
    release_camera()