import cv2
import sys
import math
import time
import random
import serial
import mediapipe as mp
import subprocess

print("RUNNING GREATER / SMALLER MODE: FINGERS ONLY. USER DOES NOT NEED TO SPEAK.")

# -------------------------------------------------
# Greater / Smaller Mode - Finger Answer Only
# -------------------------------------------------
# Flow:
# 1. User chooses GREATER or SMALLER by gesture.
#    - Thumbs Up = Greater
#    - OK Sign = Smaller
# 2. Robot shows first number for 2 seconds.
# 3. Robot resets to 0 shortly.
# 4. Robot shows second number for 2 seconds.
# 5. User answers only with fingers:
#    - GREATER: show the larger number
#    - SMALLER: show the smaller number
# 6. Correct -> feedback "GOOD JOB" and continues to next exercise.
#    Incorrect -> feedback "TRY AGAIN" and repeats the same exercise.
#
# No microphone listening is used.
# Audio output feedback uses Windows built-in speech with Microsoft Zira Desktop.
# -------------------------------------------------

# -------------------------------------------------
# Configuration
# -------------------------------------------------

ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600

ROBOT_SHOW_SECONDS = 2.0
ROBOT_GAP_SECONDS = 0.6
FEEDBACK_SECONDS = 2.2
ANSWER_STABLE_SECONDS = 0.8

MIN_ROBOT_FINGERS = 0
MAX_ROBOT_FINGERS = 5
MIN_ANSWER = 0
MAX_ANSWER = 10

# -------------------------------------------------
# Global State
# -------------------------------------------------

game_state = "CHOOSE_MODE"

selected_mode = None            # "GREATER" or "SMALLER"
first_number = 0
second_number = 0
correct_answer = 0

current_finger_count = -1
candidate_answer = None
candidate_start_time = 0

state_start_time = time.time()
last_state_change_time = 0

feedback_text = ""
feedback_color = (255, 255, 255)
last_answer_correct = False

SIMULATION_MODE = False
ser = None

# -------------------------------------------------
# Audio Feedback - Windows speaker output only
# -------------------------------------------------

def play_feedback_audio(text):
    """
    Speaks feedback aloud using Windows built-in speech with Microsoft Zira Desktop.
    This does NOT use the microphone and does NOT require the user to talk.
    It starts a fresh Windows speech process each time, so it does not get stuck.
    """
    print(f"Speaking with Microsoft Zira Desktop: {text}")

    try:
        safe_text = str(text).replace("'", "")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $speak.SelectVoice('Microsoft Zira Desktop'); } catch { "
            "  try { "
            "    $speak.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female, "
            "                               [System.Speech.Synthesis.VoiceAge]::Adult); "
            "  } catch { } "
            "}; "
            "$speak.Rate = -1; "
            "$speak.Volume = 90; "
            f"$speak.Speak('{safe_text}');"
        )

        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Could not play audio feedback: {e}")


print("Audio feedback active: Windows will say GOOD JOB / TRY AGAIN using a female voice when available")

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def limit_robot_fingers(number):
    return max(MIN_ROBOT_FINGERS, min(MAX_ROBOT_FINGERS, int(number)))


def init_serial_connection(port=ARDUINO_PORT, baud_rate=ARDUINO_BAUD_RATE):
    """
    Tries to connect to Arduino.
    If Arduino is not connected, returns None and continues in simulation mode.
    """
    global SIMULATION_MODE

    try:
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        SIMULATION_MODE = False
        print(f"Connected to Arduino on {port}")
        return arduino
    except Exception as e:
        SIMULATION_MODE = True
        print("Arduino not connected. Running in simulation mode.")
        print(f"   Serial details: {e}")
        return None


def send_robot_fingers(number):
    """
    Sends 0..5 to Arduino.
    If Arduino is missing, prints the command instead of crashing.
    """
    global ser

    safe_number = limit_robot_fingers(number)
    command = str(safe_number).encode()

    if ser is not None and ser.is_open:
        try:
            ser.write(command)
            print(f"Robot shows {safe_number} fingers")
        except Exception as e:
            print(f"Could not send command to Arduino: {e}")
            print(f"Simulation fallback - robot would show {safe_number} fingers")
            ser = None
    else:
        print(f"Simulation mode - robot would show {safe_number} fingers")


def count_fingers_single_hand(hand_landmarks):
    """
    Stable rule-based finger counter.
    Counts 0..5 for one hand.
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

    return limit_robot_fingers(sum(fingers))


def count_fingers_two_hands(all_hand_landmarks):
    """
    Counts total fingers from one or two detected hands.
    Returns -1 if no hand is detected.
    """
    if not all_hand_landmarks:
        return -1

    total = 0
    detected = 0

    for hand_landmarks in all_hand_landmarks:
        count = count_fingers_single_hand(hand_landmarks)
        if count >= 0:
            total += count
            detected += 1

    if detected == 0:
        return -1

    return max(MIN_ANSWER, min(MAX_ANSWER, total))


def is_ok_gesture(hand_landmarks):
    """
    OK sign:
    thumb tip close to index tip, while middle/ring/pinky are relatively open.
    Used for SMALLER mode selection.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return False

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    thumb_index_dist = get_dist(thumb_tip, index_tip)

    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.10

    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)

    return thumb_index_dist < hand_scale * 0.35 and middle_open and ring_open and pinky_open


def is_thumbs_up(hand_landmarks):
    """
    Thumbs up gesture for GREATER mode:
    thumb extended, other four fingers relatively closed.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return False

    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65

    def is_finger_closed(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist < 1.10

    index_closed = is_finger_closed(8, 6)
    middle_closed = is_finger_closed(12, 10)
    ring_closed = is_finger_closed(16, 14)
    pinky_closed = is_finger_closed(20, 18)

    return thumb_open and index_closed and middle_closed and ring_closed and pinky_closed


def any_ok_gesture(all_hand_landmarks):
    if not all_hand_landmarks:
        return False
    return any(is_ok_gesture(hand) for hand in all_hand_landmarks)


def any_thumbs_up(all_hand_landmarks):
    if not all_hand_landmarks:
        return False
    return any(is_thumbs_up(hand) for hand in all_hand_landmarks)


def is_back_gesture(hand_landmarks):
    """
    Back gesture: Shaka / Call Me sign.
    Thumb and pinky open, index/middle/ring closed.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return False

    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.15

    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65
    index_open = is_finger_open(8, 6)
    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)

    return thumb_open and pinky_open and not index_open and not middle_open and not ring_open


def any_back_gesture(all_hand_landmarks):
    if not all_hand_landmarks:
        return False
    return any(is_back_gesture(hand) for hand in all_hand_landmarks)


def generate_exercise(mode):
    """
    Generates two different numbers between 0 and 5.
    The robot can show only 0..5.
    Equal numbers are avoided because GREATER / SMALLER would be ambiguous.
    """
    a = random.randint(0, 5)
    b = random.randint(0, 5)

    while b == a:
        b = random.randint(0, 5)

    if mode == "GREATER":
        ans = max(a, b)
    elif mode == "SMALLER":
        ans = min(a, b)
    else:
        ans = 0

    return a, b, ans


def start_new_exercise():
    """
    Creates a new greater/smaller exercise for the selected mode.
    """
    global first_number, second_number, correct_answer
    global game_state, state_start_time, candidate_answer, candidate_start_time

    first_number, second_number, correct_answer = generate_exercise(selected_mode)
    candidate_answer = None
    candidate_start_time = 0

    print("-------------------------------------------------")
    if selected_mode == "GREATER":
        print(f"New exercise: choose GREATER from {first_number} and {second_number}. Answer: {correct_answer}")
    else:
        print(f"New exercise: choose SMALLER from {first_number} and {second_number}. Answer: {correct_answer}")

    game_state = "SHOW_FIRST"
    state_start_time = time.time()
    send_robot_fingers(first_number)


def select_mode(mode):
    """
    Selects GREATER or SMALLER and starts the first exercise.
    """
    global selected_mode, last_state_change_time

    selected_mode = mode
    last_state_change_time = time.time()

    if selected_mode == "GREATER":
        print("Mode selected: GREATER")
    else:
        print("Mode selected: SMALLER")

    start_new_exercise()


def get_mode_text():
    if selected_mode == "GREATER":
        return "GREATER"
    if selected_mode == "SMALLER":
        return "SMALLER"
    return "?"


def check_answer(answer_value, current_time):
    """
    Checks finger-only answer after it was held stable.
    """
    global feedback_text, feedback_color, last_answer_correct
    global game_state, state_start_time, last_state_change_time

    if answer_value == correct_answer:
        feedback_text = "GOOD JOB"
        feedback_color = (0, 255, 0)
        last_answer_correct = True
        play_feedback_audio("Good job")
        print("Correct answer")
    else:
        feedback_text = "TRY AGAIN"
        feedback_color = (0, 0, 255)
        last_answer_correct = False
        play_feedback_audio("Try again")
        print(f"Wrong answer: user showed {answer_value}, answer is {correct_answer}")

    game_state = "FEEDBACK"
    state_start_time = current_time
    last_state_change_time = current_time


# -------------------------------------------------
# Init Arduino
# -------------------------------------------------

ser = init_serial_connection()

# -------------------------------------------------
# Init MediaPipe
# -------------------------------------------------

try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    print("MediaPipe error: this code requires the legacy MediaPipe Solutions API.")
    print("   Recommended install inside your venv:")
    print("   python -m pip install mediapipe==0.10.14")
    raise SystemExit(1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera Error: could not open camera")
    if ser is not None and ser.is_open:
        ser.close()
    raise SystemExit(1)

print("Greater / Smaller Mode Started - finger answer only")
print("Press 'q' to quit")

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

try:
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
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

            all_hand_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else []
            current_finger_count = count_fingers_two_hands(all_hand_landmarks)
            current_ok_gesture = any_ok_gesture(all_hand_landmarks)
            current_thumbs_up = any_thumbs_up(all_hand_landmarks)
            current_back_gesture = any_back_gesture(all_hand_landmarks)

            if all_hand_landmarks:
                for hand_landmarks in all_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            current_time = time.time()
            elapsed = current_time - state_start_time

            # -------------------------------------------------
            # CHOOSE_MODE
            # -------------------------------------------------
            if game_state == "CHOOSE_MODE":
                cv2.putText(frame, "Greater / Smaller Mode", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.putText(frame, "Choose mode with gesture:", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.putText(frame, "THUMBS UP = Greater",
                            (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.putText(frame, "OK SIGN = Smaller",
                            (30, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.putText(frame, "SHAKA = Back to Education Menu",
                            (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                if SIMULATION_MODE:
                    cv2.putText(frame, "Arduino: simulation mode",
                                (30, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected total fingers: {current_finger_count}",
                                (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_robot_fingers(0)
                        print("Back gesture detected - returning to Education Menu")
                        sys.exit(10)
                    elif current_thumbs_up:
                        select_mode("GREATER")
                    elif current_ok_gesture:
                        select_mode("SMALLER")

            # -------------------------------------------------
            # SHOW_FIRST
            # -------------------------------------------------
            elif game_state == "SHOW_FIRST":
                cv2.putText(frame, "Watch the robot", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.putText(frame, f"First number: {first_number}",
                            (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                cv2.putText(frame, f"Mode: {get_mode_text()}",
                            (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if elapsed >= ROBOT_SHOW_SECONDS:
                    send_robot_fingers(0)
                    game_state = "GAP"
                    state_start_time = current_time
                    print("Robot resets between numbers")

            # -------------------------------------------------
            # GAP
            # -------------------------------------------------
            elif game_state == "GAP":
                cv2.putText(frame, "Get ready for the next number...",
                            (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if elapsed >= ROBOT_GAP_SECONDS:
                    send_robot_fingers(second_number)
                    game_state = "SHOW_SECOND"
                    state_start_time = current_time
                    print(f"Second number: {second_number}")

            # -------------------------------------------------
            # SHOW_SECOND
            # -------------------------------------------------
            elif game_state == "SHOW_SECOND":
                cv2.putText(frame, "Watch the robot", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.putText(frame, f"Second number: {second_number}",
                            (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                cv2.putText(frame, f"Question: Which is {get_mode_text()}?",
                            (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if elapsed >= ROBOT_SHOW_SECONDS:
                    send_robot_fingers(0)
                    candidate_answer = None
                    candidate_start_time = 0
                    game_state = "WAIT_FOR_ANSWER"
                    state_start_time = current_time
                    print("Waiting for user's finger answer")

            # -------------------------------------------------
            # WAIT_FOR_ANSWER
            # -------------------------------------------------
            elif game_state == "WAIT_FOR_ANSWER":
                cv2.putText(frame, f"Numbers: {first_number} and {second_number}",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.putText(frame, f"Show the {get_mode_text()} number with your fingers",
                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

                cv2.putText(frame, "Hold your answer steady",
                            (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected fingers: {current_finger_count}",
                                (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

                    if candidate_answer != current_finger_count:
                        candidate_answer = current_finger_count
                        candidate_start_time = current_time
                    else:
                        hold_time = current_time - candidate_start_time
                        cv2.putText(frame, f"Hold time: {hold_time:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s",
                                    (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                        if hold_time >= ANSWER_STABLE_SECONDS:
                            check_answer(current_finger_count, current_time)
                else:
                    candidate_answer = None
                    candidate_start_time = 0
                    cv2.putText(frame, "No hand detected",
                                (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            # -------------------------------------------------
            # FEEDBACK
            # -------------------------------------------------
            elif game_state == "FEEDBACK":
                cv2.putText(frame, feedback_text,
                            (30, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, feedback_color, 4)

                cv2.putText(frame, "Round finished...",
                            (30, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

                if elapsed >= FEEDBACK_SECONDS:
                    game_state = "ROUND_END_MENU"
                    state_start_time = current_time
                    last_state_change_time = current_time

            # -------------------------------------------------
            # ROUND_END_MENU
            # -------------------------------------------------
            elif game_state == "ROUND_END_MENU":
                cv2.putText(frame, "Round complete", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                if last_answer_correct:
                    cv2.putText(frame, "Show OPEN HAND to continue to next exercise",
                                (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Show OPEN HAND to try the same exercise again",
                                (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

                cv2.putText(frame, "Show SHAKA to return to Education Menu",
                            (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected total fingers: {current_finger_count}",
                                (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_robot_fingers(0)
                        print("Back gesture detected - returning to Education Menu")
                        sys.exit(10)

                    elif current_finger_count == 5:
                        if last_answer_correct:
                            start_new_exercise()
                        else:
                            candidate_answer = None
                            candidate_start_time = 0
                            game_state = "WAIT_FOR_ANSWER"
                            state_start_time = current_time
                            last_state_change_time = current_time
                            print("Trying the same exercise again")

            # -------------------------------------------------
            # Display
            # -------------------------------------------------
            cv2.imshow("Greater / Smaller Mode - Finger Answer Only", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    send_robot_fingers(0)

    cap.release()
    cv2.destroyAllWindows()

    if ser is not None and ser.is_open:
        ser.close()
        print("Arduino connection closed")

    print("Greater / Smaller Mode Closed")
