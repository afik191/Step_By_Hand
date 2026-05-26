import cv2
import sys
import math
import time
import random
import serial
import mediapipe as mp
import speech_recognition as sr

# -------------------------------------------------
# Configuration
# -------------------------------------------------

ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600

# If Arduino is not connected, the program will keep running
# and will only print the commands that would have been sent.
SIMULATION_MODE = False

# -------------------------------------------------
# Global Learning Mode Variables
# -------------------------------------------------

learning_state = "START"

target_number = 0              # המספר שהרובוט מציג
spoken_number = None           # המספר שהמשתמש אמר
current_finger_count = -1      # מספר האצבעות שהמשתמש מראה

round_start_time = 0
feedback_start_time = 0
last_state_change_time = 0

feedback_text = ""
feedback_color = (255, 255, 255)

# הגבלה פיזית של הרובוט
MIN_FINGERS = 0
MAX_FINGERS = 5

# מספרים שהמשתמש יכול להגיד באנגלית
NUMBER_WORDS = {
    "ZERO": 0,
    "OH": 0,

    "ONE": 1,
    "WON": 1,

    "TWO": 2,
    "TO": 2,
    "TOO": 2,

    "THREE": 3,
    "TREE": 3,

    "FOUR": 4,
    "FOR": 4,

    "FIVE": 5,
    "FIFE": 5
}


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def limit_robot_fingers(number):
    """
    Ensures the robot never receives an invalid number.
    The robot has only 5 fingers.
    """
    return max(MIN_FINGERS, min(MAX_FINGERS, int(number)))


def init_serial_connection(port=ARDUINO_PORT, baud_rate=ARDUINO_BAUD_RATE):
    """
    Tries to connect to Arduino.
    If Arduino is not connected, returns None and the program continues in simulation mode.
    """
    global SIMULATION_MODE

    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        SIMULATION_MODE = False
        print(f"Connected to Arduino on {port}")
        return ser

    except Exception as e:
        SIMULATION_MODE = True
        print(f"Arduino not connected. Running in simulation mode.")
        print(f"   Serial details: {e}")
        return None


def send_robot_fingers(ser, number):
    """
    Sends the desired number of fingers to the Arduino.
    If Arduino is not connected, prints the command instead of crashing.
    Arduino should support commands: '0', '1', '2', '3', '4', '5'
    """
    safe_number = limit_robot_fingers(number)
    command = str(safe_number).encode()

    if ser is not None and ser.is_open:
        try:
            ser.write(command)
            print(f"Robot shows {safe_number} fingers")
        except Exception as e:
            print(f"Could not send command to Arduino: {e}")
            print(f"Simulation fallback - robot would show {safe_number} fingers")
    else:
        print(f"Simulation mode - robot would show {safe_number} fingers")


def parse_number_from_voice(command):
    """
    Converts recognized voice text into a number between 0 and 5.
    """
    command = command.upper()

    # Check digits first: "1", "2", "3"...
    for digit in range(0, 6):
        if str(digit) in command:
            return digit

    # Check word variations
    for word, number in NUMBER_WORDS.items():
        if word in command:
            return number

    return None


def count_fingers_from_landmarks(hand_landmarks):
    """
    Counts how many fingers the user shows.
    This is based on your existing rule-based approach.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return -1

    fingers = []

    # Thumb
    if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, middle, ring, pinky
    tips_idx = [8, 12, 16, 20]
    mips_idx = [6, 10, 14, 18]

    for tip, mip in zip(tips_idx, mips_idx):
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip])
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip])

        if mip_dist > 0 and tip_dist / mip_dist > 1.15:
            fingers.append(1)
        else:
            fingers.append(0)

    count = sum(fingers)

    # Safety limit
    return limit_robot_fingers(count)


def is_back_gesture(hand_landmarks):
    """
    Back gesture: Shaka / Call Me sign.
    Thumb and pinky are open, while index, middle, and ring are closed.
    Used only in safe states to return to the previous menu.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return False

    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65

    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.15

    index_open = is_finger_open(8, 6)
    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)

    return thumb_open and pinky_open and not index_open and not middle_open and not ring_open


# -------------------------------------------------
# Voice Recognition Callback
# -------------------------------------------------

def voice_callback(recognizer, audio):
    global spoken_number, learning_state

    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        print(f"Heard: {command}")

        number = parse_number_from_voice(command)

        if learning_state == "WAIT_FOR_USER" and number is not None:
            spoken_number = number
            print(f"User said number: {spoken_number}")

    except sr.UnknownValueError:
        pass

    except sr.RequestError as e:
        print(f"Voice Service Error: {e}")


# -------------------------------------------------
# Init Voice Recognition
# -------------------------------------------------

r = sr.Recognizer()
m = sr.Microphone()

r.energy_threshold = 1000
r.dynamic_energy_threshold = False
r.non_speaking_duration = 0.3
r.pause_threshold = 0.3

print("Calibrating microphone...")
with m as source:
    r.adjust_for_ambient_noise(source, duration=1)

stop_listening = r.listen_in_background(
    m,
    voice_callback,
    phrase_time_limit=1.2
)

print("Voice recognition active")


# -------------------------------------------------
# Init Serial Connection
# -------------------------------------------------

ser = init_serial_connection()


# -------------------------------------------------
# Init MediaPipe
# -------------------------------------------------

# Some MediaPipe installations do not expose `solutions` directly as
# `mp.solutions`. This fallback keeps the code compatible with both cases.
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    print("MediaPipe does not expose mp.solutions directly. Using fallback imports.")
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera Error: could not open camera")
    stop_listening(wait_for_stop=False)
    if ser is not None and ser.is_open:
        ser.close()
    raise SystemExit

print("Learning Mode Started")
print("Press 'q' to quit")


# -------------------------------------------------
# Main Loop
# -------------------------------------------------

try:
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_finger_count = -1
            current_back_gesture = False

            # -----------------------------
            # Detect user's fingers
            # -----------------------------
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    current_finger_count = count_fingers_from_landmarks(hand_landmarks)
                    current_back_gesture = is_back_gesture(hand_landmarks)

            current_time = time.time()

            # -----------------------------
            # START
            # -----------------------------
            if learning_state == "START":
                cv2.putText(
                    frame,
                    "Learning Mode: Count with the Robot",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Show open hand to start",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Back gesture = return to Education Menu",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (200, 200, 200),
                    2
                )

                if SIMULATION_MODE:
                    cv2.putText(
                        frame,
                        "Arduino: simulation mode",
                        (30, 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2
                    )

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_robot_fingers(ser, 0)
                        print("Back gesture detected - returning to Education Menu")
                        sys.exit(10)

                    # מתחילים באמצעות 5 אצבעות
                    elif current_finger_count == 5:
                        learning_state = "SHOW_ROBOT"
                        last_state_change_time = current_time
                        print("Learning started")

            # -----------------------------
            # ROBOT SHOWS NUMBER
            # -----------------------------
            elif learning_state == "SHOW_ROBOT":
                target_number = random.randint(1, 5)
                spoken_number = None

                send_robot_fingers(ser, target_number)

                round_start_time = current_time
                learning_state = "WAIT_FOR_USER"

                print(f"Target number: {target_number}")

            # -----------------------------
            # WAIT FOR USER RESPONSE
            # -----------------------------
            elif learning_state == "WAIT_FOR_USER":
                cv2.putText(
                    frame,
                    f"Robot shows: {target_number}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )

                if SIMULATION_MODE:
                    cv2.putText(
                        frame,
                        "Arduino is not connected - simulation mode",
                        (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2
                    )

                cv2.putText(
                    frame,
                    "Show the same number of fingers",
                    (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "And say the number out loud",
                    (50, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

                if current_finger_count != -1:
                    cv2.putText(
                        frame,
                        f"Detected fingers: {current_finger_count}",
                        (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (200, 200, 200),
                        2
                    )

                if spoken_number is not None:
                    cv2.putText(
                        frame,
                        f"Heard number: {spoken_number}",
                        (50, 350),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (200, 200, 200),
                        2
                    )

                # בדיקה רק כשיש גם יד מזוהה וגם מספר שנאמר
                if current_finger_count != -1 and spoken_number is not None:
                    finger_correct = current_finger_count == target_number
                    voice_correct = spoken_number == target_number

                    if finger_correct and voice_correct:
                        feedback_text = "Correct! Great job!"
                        feedback_color = (0, 255, 0)

                    else:
                        mistakes = []

                        if not finger_correct:
                            mistakes.append(
                                f"You showed {current_finger_count}, but robot showed {target_number}"
                            )

                        if not voice_correct:
                            mistakes.append(
                                f"You said {spoken_number}, but robot showed {target_number}"
                            )

                        feedback_text = " | ".join(mistakes)
                        feedback_color = (0, 0, 255)

                    feedback_start_time = current_time
                    learning_state = "FEEDBACK"

            # -----------------------------
            # FEEDBACK
            # -----------------------------
            elif learning_state == "FEEDBACK":
                cv2.putText(
                    frame,
                    feedback_text,
                    (30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    feedback_color,
                    2
                )

                cv2.putText(
                    frame,
                    "Round finished",
                    (30, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                if current_time - feedback_start_time > 2:
                    learning_state = "ROUND_END_MENU"
                    last_state_change_time = current_time

            # -----------------------------
            # ROUND_END_MENU
            # -----------------------------
            elif learning_state == "ROUND_END_MENU":
                cv2.putText(
                    frame,
                    "Round finished",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "Show open hand to continue",
                    (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Back gesture = return to Education Menu",
                    (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    2
                )

                if current_finger_count != -1:
                    cv2.putText(
                        frame,
                        f"Detected fingers: {current_finger_count}",
                        (30, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (200, 200, 200),
                        2
                    )

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_robot_fingers(ser, 0)
                        print("Back gesture detected - returning to Education Menu")
                        sys.exit(10)

                    elif current_finger_count == 5:
                        learning_state = "SHOW_ROBOT"
                        last_state_change_time = current_time
                        print("Continuing to next round")

            # -----------------------------
            # Display Window
            # -----------------------------
            cv2.imshow("Learning Mode - Count with Robot", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------
    send_robot_fingers(ser, 0)

    stop_listening(wait_for_stop=False)
    cap.release()
    cv2.destroyAllWindows()

    if ser is not None and ser.is_open:
        ser.close()
        print("Arduino connection closed")

    print("Learning Mode Closed")
