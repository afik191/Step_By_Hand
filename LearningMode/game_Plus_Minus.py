import cv2
import sys
import math
import time
import random
import serial
import mediapipe as mp
import subprocess

print("RUNNING MATH MODE: FINGERS ONLY. USER DOES NOT NEED TO SPEAK.")

# -------------------------------------------------
# Math Mode: Addition / Subtraction - Finger Answer Only
# -------------------------------------------------
# Flow:
# 1. User chooses ADDITION or SUBTRACTION by gesture.
#    - Thumbs Up = Addition
#    - OK Sign = Subtraction
# 2. Robot shows first number for 2 seconds.
# 3. Robot resets to 0 shortly.
# 4. Robot shows second number for 2 seconds.
# 5. User answers only with fingers using one or two hands, 0..10.
# 6. Correct -> feedback "GOOD JOB" and continues to next exercise.
#    Incorrect -> feedback "TRY AGAIN" and repeats the same exercise.
# 7. After feedback, the system waits in ROUND_END_MENU.
#    - Back gesture returns to the Education Menu.
#    - Open hand continues to the next exercise or repeats the current one.
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

BACK_EXIT_CODE = 10
BACK_COOLDOWN_SECONDS = 1.0

# -------------------------------------------------
# Global State
# -------------------------------------------------

math_state = "CHOOSE_OPERATION"

selected_operation = None       # "ADD" or "SUB"
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
    Speaks feedback aloud using Windows built-in speech with a softer female voice when available.
    This does not use the microphone and does not require the user to talk.
    It starts a fresh Windows speech process each time, so it does not get stuck
    after the first GOOD JOB / TRY AGAIN.
    """
    print(f"Speaking with soft female voice: {text}")

    try:
        safe_text = str(text).replace("'", "")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { "
            "  $speak.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female, "
            "                             [System.Speech.Synthesis.VoiceAge]::Adult); "
            "} catch { "
            "  try { $speak.SelectVoice('Microsoft Zira Desktop'); } catch { } "
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

print("Audio feedback active: Windows will say GOOD JOB / TRY AGAIN with a softer female voice when available")

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
        print(f"Serial details: {e}")
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
    Used for SUBTRACTION in the operation selection screen.
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
    Thumbs up gesture for ADDITION:
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


def is_back_gesture(hand_landmarks):
    """
    Back gesture: Shaka / Call Me sign.
    Thumb and pinky are open, index/middle/ring are closed.
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


def any_ok_gesture(all_hand_landmarks):
    if not all_hand_landmarks:
        return False
    return any(is_ok_gesture(hand) for hand in all_hand_landmarks)


def any_thumbs_up(all_hand_landmarks):
    if not all_hand_landmarks:
        return False
    return any(is_thumbs_up(hand) for hand in all_hand_landmarks)


def any_back_gesture(all_hand_landmarks):
    if not all_hand_landmarks:
        return False
    return any(is_back_gesture(hand) for hand in all_hand_landmarks)


def return_to_previous_menu():
    """
    Stops the robot movement and exits with a special code.
    The main menu file should treat exit code 10 as BACK.
    """
    send_robot_fingers(0)
    print("Back gesture detected - returning to Education Menu")
    sys.exit(BACK_EXIT_CODE)


def generate_exercise(operation):
    """
    Generates operands.
    Robot can show only 0..5 each time.
    Answer is always 0..10.
    Subtraction is never negative.
    """
    if operation == "ADD":
        a = random.randint(0, 5)
        b = random.randint(0, 5)
        ans = a + b
        return a, b, ans

    if operation == "SUB":
        a = random.randint(0, 5)
        b = random.randint(0, a)  # guarantees no negative result
        ans = a - b
        return a, b, ans

    return 0, 0, 0


def start_new_exercise():
    """
    Creates a new exercise for the selected operation.
    """
    global first_number, second_number, correct_answer
    global math_state, state_start_time, candidate_answer, candidate_start_time

    first_number, second_number, correct_answer = generate_exercise(selected_operation)
    candidate_answer = None
    candidate_start_time = 0

    print("-------------------------------------------------")
    if selected_operation == "ADD":
        print(f"New exercise: {first_number} + {second_number} = {correct_answer}")
    else:
        print(f"New exercise: {first_number} - {second_number} = {correct_answer}")

    math_state = "SHOW_FIRST"
    state_start_time = time.time()
    send_robot_fingers(first_number)


def select_operation(operation):
    """
    Selects ADD or SUB and starts the first exercise.
    """
    global selected_operation, last_state_change_time

    selected_operation = operation
    last_state_change_time = time.time()

    if selected_operation == "ADD":
        print("Operation selected: ADDITION")
    else:
        print("Operation selected: SUBTRACTION")

    start_new_exercise()


def get_operation_symbol():
    if selected_operation == "ADD":
        return "+"
    if selected_operation == "SUB":
        return "-"
    return "?"


def check_answer(answer_value, current_time):
    """
    Checks finger-only answer after it was held stable.
    """
    global feedback_text, feedback_color, last_answer_correct
    global math_state, state_start_time, last_state_change_time

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

    math_state = "FEEDBACK"
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
    print("Recommended install inside your venv:")
    print("python -m pip install mediapipe==0.10.14")
    raise SystemExit(1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera Error: could not open camera")
    if ser is not None and ser.is_open:
        ser.close()
    raise SystemExit(1)

print("Math Mode Started - finger answer only")
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
            # CHOOSE_OPERATION
            # -------------------------------------------------
            if math_state == "CHOOSE_OPERATION":
                cv2.putText(frame, "Math Mode", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

                cv2.putText(frame, "Choose operation with gesture:", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.putText(frame, "THUMBS UP = Addition",
                            (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.putText(frame, "OK SIGN = Subtraction",
                            (30, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.putText(frame, "BACK gesture = Return to Education Menu",
                            (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

                if SIMULATION_MODE:
                    cv2.putText(frame, "Arduino: simulation mode",
                                (30, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected total fingers: {current_finger_count}",
                                (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

                if current_time - last_state_change_time > BACK_COOLDOWN_SECONDS:
                    if current_back_gesture:
                        return_to_previous_menu()
                    elif current_thumbs_up:
                        select_operation("ADD")
                    elif current_ok_gesture:
                        select_operation("SUB")

            # -------------------------------------------------
            # SHOW_FIRST
            # -------------------------------------------------
            elif math_state == "SHOW_FIRST":
                cv2.putText(frame, "Watch the robot", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.putText(frame, f"First number: {first_number}",
                            (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                cv2.putText(frame, f"Operation: {get_operation_symbol()}",
                            (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if elapsed >= ROBOT_SHOW_SECONDS:
                    send_robot_fingers(0)
                    math_state = "GAP"
                    state_start_time = current_time
                    print("Robot resets between numbers")

            # -------------------------------------------------
            # GAP
            # -------------------------------------------------
            elif math_state == "GAP":
                cv2.putText(frame, "Get ready for the next number...",
                            (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if elapsed >= ROBOT_GAP_SECONDS:
                    send_robot_fingers(second_number)
                    math_state = "SHOW_SECOND"
                    state_start_time = current_time
                    print(f"Second number: {second_number}")

            # -------------------------------------------------
            # SHOW_SECOND
            # -------------------------------------------------
            elif math_state == "SHOW_SECOND":
                cv2.putText(frame, "Watch the robot", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.putText(frame, f"Second number: {second_number}",
                            (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                cv2.putText(frame, f"Question: {first_number} {get_operation_symbol()} {second_number} = ?",
                            (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if elapsed >= ROBOT_SHOW_SECONDS:
                    send_robot_fingers(0)
                    candidate_answer = None
                    candidate_start_time = 0
                    math_state = "WAIT_FOR_ANSWER"
                    state_start_time = current_time
                    print("Waiting for user's finger answer")

            # -------------------------------------------------
            # WAIT_FOR_ANSWER
            # -------------------------------------------------
            elif math_state == "WAIT_FOR_ANSWER":
                cv2.putText(frame, f"Solve: {first_number} {get_operation_symbol()} {second_number} = ?",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.putText(frame, "Show the answer with 1 or 2 hands",
                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

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
            elif math_state == "FEEDBACK":
                cv2.putText(frame, feedback_text,
                            (30, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, feedback_color, 4)

                cv2.putText(frame, "Prepare for the end menu...",
                            (30, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

                if elapsed >= FEEDBACK_SECONDS:
                    math_state = "ROUND_END_MENU"
                    state_start_time = current_time
                    last_state_change_time = current_time
                    print("Round ended - waiting for continue or back gesture")

            # -------------------------------------------------
            # ROUND_END_MENU
            # -------------------------------------------------
            elif math_state == "ROUND_END_MENU":
                cv2.putText(frame, "Round finished",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

                cv2.putText(frame, feedback_text,
                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, feedback_color, 3)

                cv2.putText(frame, "Show OPEN HAND to continue",
                            (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.putText(frame, "Show BACK gesture to return",
                            (30, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected total fingers: {current_finger_count}",
                                (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

                if current_time - last_state_change_time > BACK_COOLDOWN_SECONDS:
                    if current_back_gesture:
                        return_to_previous_menu()

                    elif current_finger_count == 5:
                        if last_answer_correct:
                            print("Continuing to next exercise")
                            start_new_exercise()
                        else:
                            print("Repeating the same exercise")
                            candidate_answer = None
                            candidate_start_time = 0
                            math_state = "WAIT_FOR_ANSWER"
                            state_start_time = current_time
                            last_state_change_time = current_time

            # -------------------------------------------------
            # Display
            # -------------------------------------------------
            cv2.imshow("Math Mode - Addition and Subtraction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    send_robot_fingers(0)

    cap.release()
    cv2.destroyAllWindows()

    if ser is not None and ser.is_open:
        ser.close()
        print("Arduino connection closed")

    print("Math Mode Closed")