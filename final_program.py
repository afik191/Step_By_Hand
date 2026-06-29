import cv2
import sys
import math
import time
import random
import pickle
import queue
import subprocess
import threading
from pathlib import Path
from collections import deque

import mediapipe as mp

# Safely import optional dependencies so the app won't crash if they are missing
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import serial
except Exception:
    serial = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import sklearn  # noqa: F401
except Exception:
    pass


# =========================================================
# Shared voice helper
# =========================================================
class VoiceInstructions:
    # Manages text-to-speech in a background thread to keep the video feed smooth
    def __init__(self, enabled=True, rate=-1, volume=95):
        self.enabled = enabled
        self.rate = max(-10, min(10, int(rate)))
        self.volume = max(0, min(100, int(volume)))
        self._queue = queue.Queue()
        self._last_key = None
        self._generation = 0
        self._speaking = threading.Event()
        self._stopped = threading.Event()
        self._process_lock = threading.Lock()
        self._current_process = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def is_speaking(self):
        return self._speaking.is_set()

    @property
    def is_busy(self):
        """True while speech is queued, starting, or currently playing."""
        return self._speaking.is_set() or not self._queue.empty()

    def announce(self, key, text, force=False):
        # Queues a new phrase and stops whatever is currently playing
        # The key prevents repeating the same phrase constantly
        if not self.enabled or self._stopped.is_set() or not text:
            return
        text = str(text).strip()
        if not text:
            return
        if not force and key == self._last_key:
            return
        self._last_key = key
        self._generation += 1
        generation = self._generation
        self._cancel_current_speech()
        self._clear_pending()
        self._queue.put((generation, text))

    def repeat(self, text):
        if not self.enabled or self._stopped.is_set() or not text:
            return
        text = str(text).strip()
        if not text:
            return
        self._generation += 1
        generation = self._generation
        self._cancel_current_speech()
        self._clear_pending()
        self._queue.put((generation, text))

    def reset(self):
        self._last_key = None

    def stop(self):
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._generation += 1
        self._cancel_current_speech()
        self._clear_pending()
        self._queue.put(None)

    def _clear_pending(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _cancel_current_speech(self):
        # Safely terminates the background text-to-speech process
        with self._process_lock:
            process = self._current_process
            self._current_process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.8)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, text = item
            if generation != self._generation or self._stopped.is_set():
                continue
            self._speak_windows(generation, text)

    def _speak_windows(self, generation, text):
        # Uses built-in Windows PowerShell for text-to-speech
        if generation != self._generation or self._stopped.is_set():
            return
        safe_text = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $speaker.SelectVoiceByHints("
            "[System.Speech.Synthesis.VoiceGender]::Female,"
            "[System.Speech.Synthesis.VoiceAge]::Adult); } catch {}; "
            f"$speaker.Rate = {self.rate}; "
            f"$speaker.Volume = {self.volume}; "
            "$prompt = New-Object System.Speech.Synthesis.PromptBuilder; "
            "$prompt.AppendBreak([System.TimeSpan]::FromMilliseconds(350)); "
            f"$prompt.AppendText('{safe_text}'); "
            "$speaker.Speak($prompt);"
        )
        self._speaking.set()
        process = None
        try:
            process = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._process_lock:
                if generation != self._generation or self._stopped.is_set():
                    try:
                        process.terminate()
                    except Exception:
                        pass
                    return
                self._current_process = process
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None
            self._speaking.clear()


# =========================================================
# Paths / assets
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
GAME_DIR = BASE_DIR / "GameMode"

# Load the machine learning predictors for gestures
MODEL_EVEN_ODD = BASE_DIR / "models" / "hand_predictor.pkl"
MODEL_RPS = BASE_DIR / "models" / "rps_predictor.pkl" if (BASE_DIR / "models" / "rps_predictor.pkl").exists() else MODEL_EVEN_ODD


# =========================================================
# UI constants
# =========================================================
# Visual configuration and timings for the interface
PANEL_BG_COLOR = (255, 255, 255)
DIVIDER_COLOR = (170, 170, 170)
TITLE_COLOR = (0, 0, 0)
OPTION_COLOR = (0, 0, 220)
NOTE_COLOR = (60, 60, 60)
HOLD_COLOR = (100, 40, 0)
GOOD_COLOR = (0, 140, 0)
BAD_COLOR = (0, 0, 180)
INFO_COLOR = (50, 50, 50)
STATUS_COLOR = (40, 40, 40)

SELECTION_HOLD_SECONDS = 1.5
GESTURE_HOLD_SECONDS = 1.5
ANSWER_STABLE_SECONDS = 0.8
RETURN_COOLDOWN = 1.5
BACK_EXIT_CODE = 10
ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600

ROBOT_SHOW_SECONDS = 3.5
ROBOT_GAP_SECONDS = 1.0
FEEDBACK_SECONDS = 2.0
COUNTING_FEEDBACK_SECONDS = 2.2
COUNTDOWN_DURATION = 3.0
FEATURE_LOCK_BEFORE_END = 0.3
FEATURE_LOCK_TIME = COUNTDOWN_DURATION - FEATURE_LOCK_BEFORE_END
VERIFY_ACTUAL_SECONDS = 1.2

MAIN_MENU = "MAIN_MENU"
GAME_MENU = "GAME_MENU"
LEARNING_MENU = "LEARNING_MENU"
SCREEN_RPS = "SCREEN_RPS"
SCREEN_EVEN_ODD = "SCREEN_EVEN_ODD"
SCREEN_COUNTING = "SCREEN_COUNTING"
SCREEN_BIG_SMALL = "SCREEN_BIG_SMALL"
SCREEN_MATH = "SCREEN_MATH"


# =========================================================
# Shared utilities
# =========================================================
def create_split_screen(frame):
    # Splits the window into camera view and text panel
    camera_view = frame.copy()
    panel = frame.copy()
    panel[:] = PANEL_BG_COLOR
    panel_h = panel.shape[0]
    cv2.line(panel, (0, 0), (0, panel_h), DIVIDER_COLOR, 3)
    return camera_view, panel


def draw_lines(frame, lines, start_y, color, scale=0.9, thickness=2, step=35):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (30, start_y + i * step), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def limit_robot_fingers(number):
    return max(0, min(5, int(number)))


def count_fingers_single_hand(hand_landmarks):
    # Counts how many fingers are up based on landmark geometry
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0:
        return -1
    fingers = []
    fingers.append(1 if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6 else 0)
    tips_idx = [8, 12, 16, 20]
    mips_idx = [6, 10, 14, 18]
    for tip, mip in zip(tips_idx, mips_idx):
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip])
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip])
        fingers.append(1 if mip_dist > 0 and tip_dist / mip_dist > 1.15 else 0)
    return limit_robot_fingers(sum(fingers))


def count_fingers_two_hands(all_hand_landmarks):
    if not all_hand_landmarks:
        return -1
    total = 0
    detected = 0
    for hand_landmarks in all_hand_landmarks:
        c = count_fingers_single_hand(hand_landmarks)
        if c >= 0:
            total += c
            detected += 1
    if detected == 0:
        return -1
    return max(0, min(10, total))


def is_ok_gesture(hand_landmarks):
    # Detects the OK sign (thumb and index closed together)
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0:
        return False
    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])

    def is_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.10

    return thumb_index_dist < hand_scale * 0.35 and is_open(12, 10) and is_open(16, 14) and is_open(20, 18)


def is_thumbs_up(hand_landmarks):
    # Detects a thumbs up sign
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
    return thumb_open and not index_open and not middle_open and not ring_open and not pinky_open


def update_hold(desired_action, current_time):
    # Acts as a timer to ensure the user holds a gesture before triggering an action
    global hold_action, hold_start_time
    if desired_action is None:
        hold_action = None
        hold_start_time = 0.0
        return None, 0.0
    if desired_action != hold_action:
        hold_action = desired_action
        hold_start_time = current_time
        return None, 0.0
    elapsed = current_time - hold_start_time
    if elapsed >= GESTURE_HOLD_SECONDS:
        action = hold_action
        hold_action = None
        hold_start_time = 0.0
        return action, elapsed
    return None, elapsed


def draw_hold_status(frame, current_time):
    h, _, _ = frame.shape
    if hold_action:
        progress = min(GESTURE_HOLD_SECONDS, current_time - hold_start_time)
        cv2.putText(frame, f"Hold action: {hold_action}", (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
        cv2.putText(frame, f"Hold selection: {progress:.1f}s / {GESTURE_HOLD_SECONDS:.1f}s", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)


def hold_label(action_name):
    labels = {
        "go_game": "GAME MODE",
        "go_learning": "LEARNING MODE",
        "back_main": "BACK",
        "open_rps": "ROCK PAPER SCISSORS",
        "open_evenodd": "EVEN ODD",
        "open_counting": "COUNTING",
        "open_math": "PLUS MINUS",
        "open_gs": "GREATER SMALLER",
    }
    return labels.get(action_name, action_name)


# =========================================================
# Voice recognition
# =========================================================
voice_queue = deque()
stop_listening = None
voice_enabled = False
voice_guide = VoiceInstructions()


def voice_callback(recognizer, audio):
    if voice_guide.is_speaking:
        return
    try:
        # Attempt to translate the audio to text
        text = recognizer.recognize_google(audio, language="en-US").upper()
        
        # Log successful recognition cleanly to the console
        print(f"\n[🎙️ VOICE INPUT] Success: '{text}'")
        
        voice_queue.append(text)
        
    except sr.UnknownValueError:
        # Triggers if it heard sound (like a cough, background noise, or mumbling) but couldn't make out English words
        print("\n[🎙️ VOICE INPUT] Detected sound, but could not understand the words.")
        
    except sr.RequestError as e:
        # Triggers if you lose internet connection or Google API blocks the request
        print(f"\n[🎙️ VOICE INPUT] Network/API Error: {e}")
        
    except Exception as e:
        # Catch-all for any other unexpected errors
        print(f"\n[🎙️ VOICE INPUT] Unexpected Error: {e}")

def init_voice():
    # Initializes the microphone listening thread
    global stop_listening, voice_enabled
    voice_enabled = False
    stop_listening = None
    if sr is None:
        return
    try:
        r = sr.Recognizer()
        m = sr.Microphone()
        r.energy_threshold = 1000
        r.dynamic_energy_threshold = False
        r.non_speaking_duration = 0.3
        r.pause_threshold = 0.3
        with m as source:
            r.adjust_for_ambient_noise(source, duration=1)
        stop_listening = r.listen_in_background(m, voice_callback, phrase_time_limit=1.2)
        voice_enabled = True
    except Exception:
        voice_enabled = False
        stop_listening = None


def stop_voice():
    global stop_listening
    if stop_listening is not None:
        try:
            stop_listening(wait_for_stop=False)
        except Exception:
            pass
    stop_listening = None


# =========================================================
# Serial / robot
# =========================================================
SIMULATION_MODE = False
ser = None


def init_serial_connection(port=ARDUINO_PORT, baud_rate=ARDUINO_BAUD_RATE):
    # Connects to the Arduino. Falls back to simulation mode if hardware is missing.
    global SIMULATION_MODE
    if serial is None:
        SIMULATION_MODE = True
        return None
    try:
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        SIMULATION_MODE = False
        return arduino
    except Exception:
        SIMULATION_MODE = True
        return None


def send_robot_fingers(number):
    safe_number = limit_robot_fingers(number)
    if ser is not None and ser.is_open:
        try:
            ser.write(str(safe_number).encode())
        except Exception:
            pass


# =========================================================
# Models
# =========================================================
even_odd_model = None
rps_model = None


def load_models():
    global even_odd_model, rps_model
    try:
        with open(MODEL_EVEN_ODD, "rb") as f:
            even_odd_model = pickle.load(f)
    except Exception as e:
        print(f"WARNING: failed loading even-odd model: {e}")
        even_odd_model = None
    try:
        with open(MODEL_RPS, "rb") as f:
            rps_model = pickle.load(f)
    except Exception as e:
        print(f"WARNING: failed loading rps model: {e}")
        rps_model = even_odd_model


# =========================================================
# Detection helpers for predictive games
# =========================================================
def extract_prediction_features(hand_landmarks, previous_distances):
    # Extracts hand proportions and movement speeds to feed into the prediction models
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0:
        return None, previous_distances
    tips = [4, 8, 12, 16, 20]
    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
    if previous_distances is None:
        speeds = [0.0] * 5
    else:
        speeds = [cur - prv for cur, prv in zip(current_distances, previous_distances)]
    inter = [get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i + 1]]) / hand_scale for i in range(4)]
    return current_distances + speeds + inter, current_distances


# =========================================================
# Global app state
# =========================================================
# Track the current active menu or game state
current_screen = MAIN_MENU
previous_menu_screen = MAIN_MENU
hold_action = None
hold_start_time = 0.0
last_action_time = 0.0

# RPS state
ROBOT_COMMANDS = {"ROCK": 0, "PAPER": 5, "SCISSORS": 2}
WINNING_MOVE = {"ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK"}
rps_state = "READY"
rps_countdown_start = 0.0
rps_verify_start = 0.0
rps_candidate_actual_start = 0.0
rps_candidate_actual = None
rps_predicted_user_move = "UNKNOWN"
rps_actual_user_move = "UNKNOWN"
rps_robot_move = "UNKNOWN"
rps_result_text = ""
rps_result_color = INFO_COLOR
rps_confidence_percent = 0.0
rps_locked_features = None
rps_prev_distances = None
rps_has_predicted = False

# Even odd state
even_state = "CHOOSE_SIDE"
even_selected_side = None
even_countdown_start = 0.0
even_verify_start = 0.0
even_candidate_actual_start = 0.0
even_candidate_actual = None
even_predicted_user_move = -1
even_actual_user_move = -1
even_robot_move = 0
even_result_text = ""
even_result_color = INFO_COLOR
even_confidence_percent = 0.0
even_locked_features = None
even_prev_distances = None
even_has_predicted = False

# Counting state
NUMBER_WORDS = {
    "ZERO": 0, "OH": 0, "ONE": 1, "WON": 1, "TWO": 2, "TO": 2, "TOO": 2,
    "THREE": 3, "TREE": 3, "FOUR": 4, "FOR": 4, "FIVE": 5, "FIFE": 5,
}
count_state = "READY"
count_target_number = 0
count_spoken_number = None
count_heard_text = ""
count_candidate_fingers = None
count_candidate_start_time = 0.0
count_stable_finger_answer = None
count_feedback_text = ""
count_feedback_color = INFO_COLOR
count_feedback_start_time = 0.0

# Big/small state
gs_state = "CHOOSE_MODE"
gs_selected_mode = None
gs_first_number = 0
gs_second_number = 0
gs_correct_answer = 0
gs_candidate_answer = None
gs_candidate_start_time = 0.0
gs_state_start_time = 0.0
gs_feedback_text = ""
gs_feedback_color = INFO_COLOR
gs_last_answer_correct = False
gs_current_level = 1
gs_consecutive_correct = 0
gs_consecutive_wrong = 0

# Math state
math_state = "CHOOSE_OPERATION"
math_selected_operation = None
math_first_number = 0
math_second_number = 0
math_correct_answer = 0
math_candidate_answer = None
math_candidate_start_time = 0.0
math_state_start_time = 0.0
math_feedback_text = ""
math_feedback_color = INFO_COLOR
math_last_answer_correct = False
math_current_level = 1
math_consecutive_correct = 0
math_consecutive_wrong = 0

REQUIRED_TO_LEVEL_UP = 2
REQUIRED_TO_LEVEL_DOWN = 2
MAX_LEVEL = 4


# =========================================================
# Screen transitions
# =========================================================
def set_screen(screen):
    global current_screen, previous_menu_screen, hold_action, hold_start_time, last_action_time
    current_screen = screen
    hold_action = None
    hold_start_time = 0.0
    last_action_time = time.time() + 0.2
    voice_guide.reset()


def back_to_previous_menu():
    if current_screen in [SCREEN_RPS, SCREEN_EVEN_ODD]:
        set_screen(GAME_MENU)
    elif current_screen in [SCREEN_COUNTING, SCREEN_BIG_SMALL, SCREEN_MATH]:
        set_screen(LEARNING_MENU)
    else:
        set_screen(MAIN_MENU)
    send_robot_fingers(0)


# =========================================================
# Game-specific helpers
# =========================================================
def parse_number_from_voice(command):
    # Extracts spoken digits or number words from voice commands
    command = command.upper()
    for digit in range(0, 6):
        if str(digit) in command:
            return digit
    for word, number in NUMBER_WORDS.items():
        if word in command:
            return number
    return None


def classify_rps_from_fingers(finger_count):
    if finger_count == 0:
        return "ROCK"
    if finger_count == 2:
        return "SCISSORS"
    if finger_count == 5:
        return "PAPER"
    return "UNKNOWN"


def map_model_output_to_rps(raw_prediction):
    try:
        pred = int(raw_prediction)
    except Exception:
        return "UNKNOWN"
    if pred == 0:
        return "ROCK"
    if pred == 2:
        return "SCISSORS"
    if pred == 5:
        return "PAPER"
    if pred <= 1:
        return "ROCK"
    if pred in [2, 3]:
        return "SCISSORS"
    return "PAPER"


def calculate_rps_result(actual_move, robot_choice):
    if actual_move == "UNKNOWN":
        return "No clear final move", NOTE_COLOR
    if actual_move == robot_choice:
        return "TIE", NOTE_COLOR
    if WINNING_MOVE[actual_move] == robot_choice:
        return "ROBOT WINS", GOOD_COLOR
    return "USER WINS", BAD_COLOR


def calculate_even_result(actual_move, robot_choice, selected_side):
    if actual_move < 0:
        return "No clear final hand", NOTE_COLOR
    total = actual_move + robot_choice
    user_wins = (selected_side == "EVEN" and total % 2 == 0) or (selected_side == "ODD" and total % 2 != 0)
    if user_wins:
        return "USER WINS", BAD_COLOR
    return "ROBOT WINS", GOOD_COLOR


# =========================================================
# Reset/start functions
# =========================================================
def rps_start_round():
    global rps_state, rps_countdown_start, rps_verify_start, rps_candidate_actual, rps_candidate_actual_start
    global rps_predicted_user_move, rps_actual_user_move, rps_robot_move, rps_result_text, rps_result_color
    global rps_confidence_percent, rps_locked_features, rps_has_predicted, rps_prev_distances
    rps_state = "COUNTDOWN"
    rps_countdown_start = time.time()
    rps_verify_start = 0.0
    rps_candidate_actual_start = 0.0
    rps_candidate_actual = None
    rps_predicted_user_move = "UNKNOWN"
    rps_actual_user_move = "UNKNOWN"
    rps_robot_move = "UNKNOWN"
    rps_result_text = ""
    rps_result_color = INFO_COLOR
    rps_confidence_percent = 0.0
    rps_locked_features = None
    rps_prev_distances = None
    rps_has_predicted = False


def rps_finish_result():
    global rps_state, rps_result_text, rps_result_color
    rps_result_text, rps_result_color = calculate_rps_result(rps_actual_user_move, rps_robot_move)
    rps_state = "RESULT"


def even_start_round():
    global even_state, even_countdown_start, even_verify_start, even_candidate_actual, even_candidate_actual_start
    global even_predicted_user_move, even_actual_user_move, even_robot_move, even_result_text, even_result_color
    global even_confidence_percent, even_locked_features, even_has_predicted, even_prev_distances
    even_state = "COUNTDOWN"
    even_countdown_start = time.time()
    even_verify_start = 0.0
    even_candidate_actual_start = 0.0
    even_candidate_actual = None
    even_predicted_user_move = -1
    even_actual_user_move = -1
    even_robot_move = 0
    even_result_text = ""
    even_result_color = INFO_COLOR
    even_confidence_percent = 0.0
    even_locked_features = None
    even_prev_distances = None
    even_has_predicted = False


def even_finish_result():
    global even_state, even_result_text, even_result_color
    even_result_text, even_result_color = calculate_even_result(even_actual_user_move, even_robot_move, even_selected_side)
    even_state = "RESULT"


def count_start_round():
    global count_state, count_target_number, count_spoken_number, count_heard_text
    global count_candidate_fingers, count_candidate_start_time, count_stable_finger_answer
    global count_feedback_text, count_feedback_color
    count_target_number = random.randint(1, 5)
    count_spoken_number = None
    count_heard_text = ""
    count_candidate_fingers = None
    count_candidate_start_time = 0.0
    count_stable_finger_answer = None
    count_feedback_text = ""
    count_feedback_color = INFO_COLOR
    send_robot_fingers(count_target_number)
    count_state = "WAIT_FOR_USER"


def count_go_to_feedback(finger_answer, voice_answer):
    global count_state, count_feedback_text, count_feedback_color, count_feedback_start_time
    finger_correct = finger_answer == count_target_number
    voice_correct = voice_answer == count_target_number if voice_enabled and voice_answer is not None else True
    if finger_correct and voice_correct:
        count_feedback_text = "GOOD JOB"
        count_feedback_color = GOOD_COLOR
    else:
        mistakes = []
        if not finger_correct:
            mistakes.append(f"Fingers: {finger_answer}")
        if not voice_correct:
            mistakes.append(f"Voice: {voice_answer}")
        count_feedback_text = "TRY AGAIN - " + " | ".join(mistakes)
        count_feedback_color = BAD_COLOR
    send_robot_fingers(0)
    count_feedback_start_time = time.time()
    count_state = "FEEDBACK"


def gs_generate_exercise(mode):
    max_val = min(5, 1 + gs_current_level)
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    while b == a:
        b = random.randint(0, max_val)
    ans = max(a, b) if mode == "GREATER" else min(a, b)
    return a, b, ans


def gs_start_new_exercise():
    global gs_first_number, gs_second_number, gs_correct_answer, gs_state, gs_state_start_time, gs_candidate_answer, gs_candidate_start_time
    gs_first_number, gs_second_number, gs_correct_answer = gs_generate_exercise(gs_selected_mode)
    gs_candidate_answer = None
    gs_candidate_start_time = 0.0
    gs_state = "SHOW_FIRST"
    gs_state_start_time = time.time()
    send_robot_fingers(gs_first_number)


def gs_go_to_mode_menu():
    global gs_state, gs_selected_mode, gs_candidate_answer, gs_candidate_start_time
    send_robot_fingers(0)
    gs_state = "CHOOSE_MODE"
    gs_selected_mode = None
    gs_candidate_answer = None
    gs_candidate_start_time = 0.0


def gs_check_answer(answer_value):
    # Handles dynamic difficulty by updating the level based on correct or wrong answers
    global gs_feedback_text, gs_feedback_color, gs_last_answer_correct, gs_state, gs_state_start_time
    global gs_current_level, gs_consecutive_correct, gs_consecutive_wrong
    if answer_value == gs_correct_answer:
        gs_consecutive_correct += 1
        gs_consecutive_wrong = 0
        gs_last_answer_correct = True
        gs_feedback_color = GOOD_COLOR
        if gs_consecutive_correct >= REQUIRED_TO_LEVEL_UP:
            if gs_current_level < MAX_LEVEL:
                gs_current_level += 1
                gs_feedback_text = "CORRECT! LEVEL UP"
            else:
                gs_feedback_text = "CORRECT! MAX LEVEL"
            gs_consecutive_correct = 0
        else:
            gs_feedback_text = "CORRECT"
    else:
        gs_consecutive_wrong += 1
        gs_consecutive_correct = 0
        gs_last_answer_correct = False
        gs_feedback_color = BAD_COLOR
        if gs_consecutive_wrong >= REQUIRED_TO_LEVEL_DOWN:
            if gs_current_level > 1:
                gs_current_level -= 1
                gs_feedback_text = "INCORRECT. LEVEL DOWN"
            else:
                gs_feedback_text = f"INCORRECT. ANSWER WAS {gs_correct_answer}"
            gs_consecutive_wrong = 0
        else:
            gs_feedback_text = f"INCORRECT. ANSWER WAS {gs_correct_answer}"
    gs_state = "FEEDBACK"
    gs_state_start_time = time.time()


def math_generate_exercise(operation):
    max_val = min(5, 1 + math_current_level)
    if operation == "ADD":
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        return a, b, a + b
    a = random.randint(0, max_val)
    b = random.randint(0, a)
    return a, b, a - b


def math_get_operation_symbol():
    return "+" if math_selected_operation == "ADD" else "-"


def math_start_new_exercise():
    global math_first_number, math_second_number, math_correct_answer, math_state, math_state_start_time, math_candidate_answer, math_candidate_start_time
    math_first_number, math_second_number, math_correct_answer = math_generate_exercise(math_selected_operation)
    math_candidate_answer = None
    math_candidate_start_time = 0.0
    math_state = "SHOW_FIRST"
    math_state_start_time = time.time()
    send_robot_fingers(math_first_number)


def math_go_to_operation_menu():
    global math_state, math_selected_operation, math_candidate_answer, math_candidate_start_time
    send_robot_fingers(0)
    math_state = "CHOOSE_OPERATION"
    math_selected_operation = None
    math_candidate_answer = None
    math_candidate_start_time = 0.0


def math_check_answer(answer_value):
    global math_feedback_text, math_feedback_color, math_last_answer_correct, math_state, math_state_start_time
    global math_current_level, math_consecutive_correct, math_consecutive_wrong
    if answer_value == math_correct_answer:
        math_consecutive_correct += 1
        math_consecutive_wrong = 0
        math_last_answer_correct = True
        math_feedback_color = GOOD_COLOR
        if math_consecutive_correct >= REQUIRED_TO_LEVEL_UP:
            if math_current_level < MAX_LEVEL:
                math_current_level += 1
                math_feedback_text = "CORRECT! LEVEL UP"
            else:
                math_feedback_text = "CORRECT! MAX LEVEL"
            math_consecutive_correct = 0
        else:
            math_feedback_text = "CORRECT"
    else:
        math_consecutive_wrong += 1
        math_consecutive_correct = 0
        math_last_answer_correct = False
        math_feedback_color = BAD_COLOR
        if math_consecutive_wrong >= REQUIRED_TO_LEVEL_DOWN:
            if math_current_level > 1:
                math_current_level -= 1
                math_feedback_text = "INCORRECT. LEVEL DOWN"
            else:
                math_feedback_text = f"INCORRECT. ANSWER WAS {math_correct_answer}"
            math_consecutive_wrong = 0
        else:
            math_feedback_text = f"INCORRECT. ANSWER WAS {math_correct_answer}"
    math_state = "FEEDBACK"
    math_state_start_time = time.time()


# =========================================================
# Voice instruction content
# =========================================================
def get_voice_payload():
    # Prepares the dynamic text string to be read aloud based on the current screen
    if current_screen == MAIN_MENU:
        return (
            (current_screen,),
            "Main menu. Show one finger for game mode, or two fingers for learning mode. You can also say game or learning.",
        )
    if current_screen == GAME_MENU:
        return (
            (current_screen,),
            "Game mode. Show one finger for rock paper scissors, or two fingers for even odd. Make an O K sign, or say back, to return to the main menu.",
        )
    if current_screen == LEARNING_MENU:
        return (
            (current_screen,),
            "Learning mode. Show one finger for counting and imitation, two fingers for addition and subtraction, or three fingers for greater and smaller. Make an O K sign, or say back, to return.",
        )
    if current_screen == SCREEN_RPS:
        if rps_state == "READY":
            return ((current_screen, rps_state), "Rock paper scissors. Show thumbs up, or say start, to begin. Rock is a closed hand, paper is five fingers, and scissors is two fingers. Say back to return.")
        if rps_state == "COUNTDOWN":
            return ((current_screen, rps_state), "Rock paper scissors. Get ready. Show your rock, paper, or scissors move before the countdown ends.")
        if rps_state == "VERIFY_ACTUAL":
            return ((current_screen, rps_state, rps_robot_move), f"Rock paper scissors. The robot played {rps_robot_move.lower()}. Hold your final move steady.")
        if rps_state == "RESULT":
            return ((current_screen, rps_state, rps_result_text, rps_robot_move, rps_actual_user_move), f"Rock paper scissors result. {rps_result_text}. The robot played {rps_robot_move.lower()}, and your move was {rps_actual_user_move.lower()}. Show thumbs up, or say start, to play again. Say back to return.")
    if current_screen == SCREEN_EVEN_ODD:
        if even_state == "CHOOSE_SIDE":
            return ((current_screen, even_state), "Choose your side. Show two fingers for even, or one finger for odd. You can also say even or odd. Say back to return.")
        if even_state == "READY":
            return ((current_screen, even_state, even_selected_side), f"Even odd game. You selected {even_selected_side.lower()}. Show thumbs up, or say start, to begin. Say back to return.")
        if even_state == "COUNTDOWN":
            return ((current_screen, even_state), "Even odd game. Get ready. Show a number from zero to five before the countdown ends.")
        if even_state == "VERIFY_ACTUAL":
            return ((current_screen, even_state, even_robot_move), f"Even odd game. The robot showed {even_robot_move}. Hold your final number steady.")
        if even_state == "RESULT":
            return ((current_screen, even_state, even_result_text, even_actual_user_move, even_robot_move), f"Even odd result. {even_result_text}. Your number was {even_actual_user_move}, the robot showed {even_robot_move}. Show thumbs up, or say start, to play again. Say back to return.")
    if current_screen == SCREEN_COUNTING:
        if count_state == "READY":
            return ((current_screen, count_state), "Counting and imitation. Show thumbs up, or say start, to begin. Make an O K sign, or say back, to return.")
        if count_state == "WAIT_FOR_USER":
            return ((current_screen, count_state, count_target_number), f"Counting and imitation. The robot is showing {count_target_number}. Show the same number with your fingers and say the number out loud.")
        if count_state == "FEEDBACK":
            return ((current_screen, count_state, count_feedback_text), count_feedback_text.replace("GOOD JOB", "Good job").replace("TRY AGAIN", "Try again"))
        if count_state == "ROUND_END_MENU":
            return ((current_screen, count_state, count_feedback_text), "Round finished. Show thumbs up, or say start, to play again. Say back to return.")
    if current_screen == SCREEN_BIG_SMALL:
        if gs_state == "CHOOSE_MODE":
            return ((current_screen, gs_state), "Greater and smaller mode. Show one finger for greater, or two fingers for smaller. Say greater or smaller. Say back to return.")
        if gs_state == "READY":
            return ((current_screen, gs_state, gs_selected_mode), f"Greater and smaller mode. You selected {gs_selected_mode.lower()}. Show thumbs up, or say start, to begin. Say back to choose again.")
        if gs_state == "SHOW_FIRST":
            return ((current_screen, gs_state, gs_first_number), f"First number is {gs_first_number}.")
        if gs_state == "GAP":
            return ((current_screen, gs_state), "Greater and smaller mode. Get ready for the second number.")
        if gs_state == "SHOW_SECOND":
            return ((current_screen, gs_state, gs_second_number, gs_selected_mode), f"Second number is {gs_second_number}. Now show the {gs_selected_mode.lower()} number.")
        if gs_state == "WAIT_FOR_ANSWER":
            return ((current_screen, gs_state, gs_first_number, gs_second_number, gs_selected_mode), f"Greater and smaller mode. The numbers are {gs_first_number} and {gs_second_number}. Show the {gs_selected_mode.lower()} number with your hands and hold it steady.")
        if gs_state == "FEEDBACK":
            return ((current_screen, gs_state, gs_feedback_text), gs_feedback_text.replace("ANSWER WAS", "The answer was"))
        if gs_state == "ROUND_END_MENU":
            action = "the next exercise" if gs_last_answer_correct else "try again"
            return ((current_screen, gs_state, gs_feedback_text, gs_last_answer_correct), f"Greater and smaller mode. Round finished. Show thumbs up, or say start, for {action}. Say back to return.")
    if current_screen == SCREEN_MATH:
        if math_state == "CHOOSE_OPERATION":
            return ((current_screen, math_state), "Addition and subtraction. Show one finger for addition, or two fingers for subtraction. Say plus or minus. Say back to return.")
        if math_state == "READY":
            operation_name = "addition" if math_selected_operation == "ADD" else "subtraction"
            return ((current_screen, math_state, math_selected_operation), f"Addition and subtraction. You selected {operation_name}. Show thumbs up, or say start, to begin. Say back to choose again.")
        if math_state == "SHOW_FIRST":
            return ((current_screen, math_state, math_first_number), f"First number is {math_first_number}.")
        if math_state == "GAP":
            return ((current_screen, math_state), "Addition and subtraction. Get ready for the second number.")
        if math_state == "SHOW_SECOND":
            operation_word = "plus" if math_selected_operation == "ADD" else "minus"
            return ((current_screen, math_state, math_first_number, math_second_number, math_selected_operation), f"Second number is {math_second_number}. The exercise is {math_first_number} {operation_word} {math_second_number}.")
        if math_state == "WAIT_FOR_ANSWER":
            operation_word = "plus" if math_selected_operation == "ADD" else "minus"
            return ((current_screen, math_state, math_first_number, math_second_number, math_selected_operation), f"Addition and subtraction. Solve {math_first_number} {operation_word} {math_second_number}. Show the answer with your hands and hold it steady.")
        if math_state == "FEEDBACK":
            return ((current_screen, math_state, math_feedback_text), math_feedback_text.replace("ANSWER WAS", "The answer was"))
        if math_state == "ROUND_END_MENU":
            action = "continue to the next exercise" if math_last_answer_correct else "try the exercise again"
            return ((current_screen, math_state, math_feedback_text, math_last_answer_correct), f"Addition and subtraction. Round finished. Show thumbs up, or say start, to {action}. Say back to return.")
    return (("NONE",), "")


# =========================================================
# Voice command routing
# =========================================================
def handle_voice_command(command):
    # Processes spoken commands to navigate the interface or perform actions
    global current_screen, last_action_time
    global even_selected_side, even_state
    global count_spoken_number, count_heard_text
    global gs_selected_mode, gs_state, gs_candidate_answer, gs_candidate_start_time, gs_state_start_time
    global math_selected_operation, math_state, math_candidate_answer, math_candidate_start_time, math_state_start_time

    if not command:
        return

    if current_screen == MAIN_MENU:
        if "GAME" in command:
            set_screen(GAME_MENU)
        elif "EDUCATION" in command or "LEARNING" in command:
            set_screen(LEARNING_MENU)
        return

    if current_screen == GAME_MENU:
        if "BACK" in command:
            set_screen(MAIN_MENU)
        elif "ROCK" in command or "SCISSORS" in command:
            rps_state_reset_to_ready()
            set_screen(SCREEN_RPS)
        elif "EVEN" in command or "ODD" in command:
            even_state_reset_to_choose()
            set_screen(SCREEN_EVEN_ODD)
        return

    if current_screen == LEARNING_MENU:
        if "BACK" in command:
            set_screen(MAIN_MENU)
        elif "COUNT" in command or "IMITATION" in command:
            count_state_reset_to_ready()
            set_screen(SCREEN_COUNTING)
        elif "MATH" in command or "PLUS" in command or "MINUS" in command or "ADD" in command or "SUB" in command:
            math_state_reset_to_choose()
            set_screen(SCREEN_MATH)
        elif "GREATER" in command or "SMALLER" in command or "BIGGER" in command:
            gs_state_reset_to_choose()
            set_screen(SCREEN_BIG_SMALL)
        return

    # Game voice commands
    if current_screen == SCREEN_RPS:
        if "BACK" in command:
            back_to_previous_menu()
        elif any(word in command for word in ["START", "BEGIN", "PLAY", "AGAIN"]) and rps_state in ["READY", "RESULT"]:
            send_robot_fingers(0)
            rps_start_round()
        return

    if current_screen == SCREEN_EVEN_ODD:
        if "BACK" in command:
            if even_state == "CHOOSE_SIDE":
                back_to_previous_menu()
            else:
                even_state_reset_to_choose()
        elif even_state == "CHOOSE_SIDE":
            if "EVEN" in command:
                even_selected_side = "EVEN"
                even_state = "READY"
            elif "ODD" in command:
                even_selected_side = "ODD"
                even_state = "READY"
        elif any(word in command for word in ["START", "BEGIN", "PLAY", "AGAIN"]) and even_state in ["READY", "RESULT"]:
            send_robot_fingers(0)
            even_start_round()
        return

    if current_screen == SCREEN_COUNTING:
        count_heard_text = command
        if "BACK" in command:
            back_to_previous_menu()
        elif count_state in ["READY", "ROUND_END_MENU"] and any(word in command for word in ["START", "BEGIN", "PLAY", "AGAIN"]):
            count_start_round()
        elif count_state == "WAIT_FOR_USER":
            number = parse_number_from_voice(command)
            if number is not None:
                count_spoken_number = number
        return

    if current_screen == SCREEN_BIG_SMALL:
        if "BACK" in command:
            if gs_state == "CHOOSE_MODE":
                back_to_previous_menu()
            else:
                gs_go_to_mode_menu()
        elif gs_state == "CHOOSE_MODE":
            if "GREATER" in command or "BIGGER" in command:
                gs_selected_mode = "GREATER"
                gs_state = "READY"
            elif "SMALLER" in command:
                gs_selected_mode = "SMALLER"
                gs_state = "READY"
        elif any(word in command for word in ["START", "BEGIN", "AGAIN"]) and gs_state in ["READY", "ROUND_END_MENU"]:
            if gs_state == "READY":
                gs_start_new_exercise()
            else:
                if gs_last_answer_correct:
                    gs_start_new_exercise()
                else:
                    gs_candidate_answer = None
                    gs_candidate_start_time = 0.0
                    gs_state = "WAIT_FOR_ANSWER"
                    gs_state_start_time = time.time()
        return

    if current_screen == SCREEN_MATH:
        if "BACK" in command:
            if math_state == "CHOOSE_OPERATION":
                back_to_previous_menu()
            else:
                math_go_to_operation_menu()
        elif math_state == "CHOOSE_OPERATION":
            if "ADD" in command or "PLUS" in command:
                math_selected_operation = "ADD"
                math_state = "READY"
            elif "SUB" in command or "MINUS" in command:
                math_selected_operation = "SUB"
                math_state = "READY"
        elif any(word in command for word in ["START", "BEGIN", "AGAIN"]) and math_state in ["READY", "ROUND_END_MENU"]:
            if math_state == "READY":
                math_start_new_exercise()
            else:
                if math_last_answer_correct:
                    math_start_new_exercise()
                else:
                    math_candidate_answer = None
                    math_candidate_start_time = 0.0
                    math_state = "WAIT_FOR_ANSWER"
                    math_state_start_time = time.time()
        return


# =========================================================
# Reset wrappers for entering screens
# =========================================================
def rps_state_reset_to_ready():
    global rps_state, rps_prev_distances
    send_robot_fingers(0)
    rps_state = "READY"
    rps_prev_distances = None


def even_state_reset_to_choose():
    global even_state, even_selected_side, even_prev_distances
    send_robot_fingers(0)
    even_state = "CHOOSE_SIDE"
    even_selected_side = None
    even_prev_distances = None


def count_state_reset_to_ready():
    global count_state
    send_robot_fingers(0)
    count_state = "READY"


def gs_state_reset_to_choose():
    global gs_state, gs_selected_mode
    send_robot_fingers(0)
    gs_state = "CHOOSE_MODE"
    gs_selected_mode = None


def math_state_reset_to_choose():
    global math_state, math_selected_operation
    send_robot_fingers(0)
    math_state = "CHOOSE_OPERATION"
    math_selected_operation = None


# =========================================================
# Render/update functions per screen
# =========================================================
def render_main_menu(panel, current_time, current_fingers, ok_detected):
    global current_screen
    draw_lines(panel, ["MAIN MENU"], 50, TITLE_COLOR, scale=1.2, thickness=3)
    draw_lines(panel, ["Show 1 finger = Game Mode", "Show 2 fingers = Learning Mode"], 110, OPTION_COLOR, scale=0.85, thickness=2, step=40)
    draw_lines(panel, ["Voice: say GAME or LEARNING", "OK sign = Back (in sub menus)"], 210, NOTE_COLOR, scale=0.7, thickness=2, step=35)

    desired_action = None
    if current_time > last_action_time:
        if current_fingers == 1:
            desired_action = "go_game"
        elif current_fingers == 2:
            desired_action = "go_learning"
    action, _ = update_hold(desired_action, current_time)
    if action == "go_game":
        set_screen(GAME_MENU)
    elif action == "go_learning":
        set_screen(LEARNING_MENU)


def render_game_menu(panel, current_time, current_fingers, ok_detected):
    draw_lines(panel, ["GAME MODE"], 50, TITLE_COLOR, scale=1.2, thickness=3)
    draw_lines(panel, ["Show 1 finger = Rock Paper Scissors", "Show 2 fingers = Even Odd"], 110, OPTION_COLOR, scale=0.8, thickness=2, step=40)
    draw_lines(panel, ["OK sign = Back to Main Menu", "Voice: ROCK / EVEN / BACK"], 210, NOTE_COLOR, scale=0.7, thickness=2, step=35)

    desired_action = None
    if current_time > last_action_time:
        if ok_detected:
            desired_action = "back_main"
        elif current_fingers == 1:
            desired_action = "open_rps"
        elif current_fingers == 2:
            desired_action = "open_evenodd"
    action, _ = update_hold(desired_action, current_time)
    if action == "back_main":
        set_screen(MAIN_MENU)
    elif action == "open_rps":
        rps_state_reset_to_ready()
        set_screen(SCREEN_RPS)
    elif action == "open_evenodd":
        even_state_reset_to_choose()
        set_screen(SCREEN_EVEN_ODD)


def render_learning_menu(panel, current_time, current_fingers, ok_detected):
    draw_lines(panel, ["LEARNING MODE"], 50, TITLE_COLOR, scale=1.2, thickness=3)
    draw_lines(panel, ["Show 1 finger = Counting / Imitation", "Show 2 fingers = Addition / Subtraction", "Show 3 fingers = Greater / Smaller"], 100, OPTION_COLOR, scale=0.75, thickness=2, step=35)
    draw_lines(panel, ["OK sign = Back to Main Menu", "Voice: COUNTING / MATH / GREATER / BACK"], 220, NOTE_COLOR, scale=0.65, thickness=2, step=30)

    desired_action = None
    if current_time > last_action_time:
        if ok_detected:
            desired_action = "back_main"
        elif current_fingers == 1:
            desired_action = "open_counting"
        elif current_fingers == 2:
            desired_action = "open_math"
        elif current_fingers == 3:
            desired_action = "open_gs"
    action, _ = update_hold(desired_action, current_time)
    if action == "back_main":
        set_screen(MAIN_MENU)
    elif action == "open_counting":
        count_state_reset_to_ready()
        set_screen(SCREEN_COUNTING)
    elif action == "open_math":
        math_state_reset_to_choose()
        set_screen(SCREEN_MATH)
    elif action == "open_gs":
        gs_state_reset_to_choose()
        set_screen(SCREEN_BIG_SMALL)


def render_rps(panel, camera_view, current_time, hand_landmarks_list, first_hand_landmarks, current_fingers_single, current_ok, current_thumbs, h, w):
    # Handles logic and UI updates for Rock Paper Scissors mode
    global rps_state, rps_countdown_start, rps_candidate_actual, rps_candidate_actual_start, rps_actual_user_move
    global rps_prev_distances, rps_locked_features, rps_has_predicted, rps_verify_start, rps_robot_move, rps_predicted_user_move, rps_confidence_percent

    current_rps_gesture = "UNKNOWN"
    current_features = None
    if first_hand_landmarks is not None:
        current_rps_gesture = classify_rps_from_fingers(current_fingers_single)
        current_features, rps_prev_distances = extract_prediction_features(first_hand_landmarks, rps_prev_distances)
    else:
        rps_prev_distances = None

    if rps_state == "READY":
        draw_lines(panel, ["ROCK PAPER SCISSORS"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, ["Thumbs up = Start game", "Voice: say START"], 90, OPTION_COLOR, scale=0.75, thickness=2, step=30)
        draw_lines(panel, ["OK sign = Back", "Voice: BACK", "Rock=0 | Paper=5 | Scissors=2"], 170, NOTE_COLOR, scale=0.65, thickness=2, step=25)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            rps_start_round()
        elif action == "BACK":
            back_to_previous_menu()

    elif rps_state == "COUNTDOWN":
        elapsed = current_time - rps_countdown_start
        count = int(math.ceil(max(0.0, COUNTDOWN_DURATION - elapsed)))
        draw_lines(panel, ["ROCK PAPER SCISSORS"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, ["Show your move now"], 80, OPTION_COLOR, scale=0.75, thickness=2)
        draw_lines(panel, ["Robot predicts before end"], 120, NOTE_COLOR, scale=0.65, thickness=2)
        if count > 0:
            cv2.putText(camera_view, str(count), (w // 2 - 45, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 4.6, OPTION_COLOR, 8)
        if current_rps_gesture != "UNKNOWN":
            cv2.putText(panel, f"Detected: {current_rps_gesture}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
        draw_lines(panel, ["OK sign = Back"], 230, NOTE_COLOR, scale=0.65, thickness=2)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            back_to_previous_menu()
            return
        if elapsed >= FEATURE_LOCK_TIME and rps_locked_features is None and current_features is not None:
            rps_locked_features = list(current_features)
        if elapsed >= COUNTDOWN_DURATION and not rps_has_predicted:
            if rps_locked_features is not None and np is not None and rps_model is not None:
                probs = rps_model.predict_proba([rps_locked_features])[0]
                raw = int(np.argmax(probs))
                rps_confidence_percent = float(probs[raw] * 100)
                rps_predicted_user_move = map_model_output_to_rps(raw)
                rps_robot_move = WINNING_MOVE.get(rps_predicted_user_move, random.choice(["ROCK", "PAPER", "SCISSORS"]))
            else:
                rps_predicted_user_move = "UNKNOWN"
                rps_robot_move = random.choice(["ROCK", "PAPER", "SCISSORS"])
            send_robot_fingers(ROBOT_COMMANDS[rps_robot_move])
            rps_has_predicted = True
            rps_state = "VERIFY_ACTUAL"
            rps_verify_start = current_time
            rps_candidate_actual = None

    elif rps_state == "VERIFY_ACTUAL":
        elapsed = current_time - rps_verify_start
        draw_lines(panel, ["FINAL MOVE CHECK"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Robot played: {rps_robot_move}", "Hold real move now"], 80, OPTION_COLOR, scale=0.7, thickness=2, step=30)
        draw_lines(panel, [f"Predicted: {rps_predicted_user_move} ({rps_confidence_percent:.1f}%)"], 150, NOTE_COLOR, scale=0.65, thickness=2)
        if current_rps_gesture != "UNKNOWN":
            cv2.putText(panel, f"Detected: {current_rps_gesture}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
            if rps_candidate_actual != current_rps_gesture:
                rps_candidate_actual = current_rps_gesture
                rps_candidate_actual_start = current_time
            else:
                hold = current_time - rps_candidate_actual_start
                cv2.putText(panel, f"Stable hold: {hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, HOLD_COLOR, 2)
                if hold >= ANSWER_STABLE_SECONDS:
                    rps_actual_user_move = current_rps_gesture
                    rps_finish_result()
        else:
            rps_candidate_actual = None
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            back_to_previous_menu()
            return
        if elapsed >= VERIFY_ACTUAL_SECONDS and rps_state == "VERIFY_ACTUAL":
            if current_rps_gesture != "UNKNOWN":
                rps_actual_user_move = current_rps_gesture
            rps_finish_result()

    elif rps_state == "RESULT":
        draw_lines(panel, ["RESULT"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Predicted: {rps_predicted_user_move}", f"Actual: {rps_actual_user_move}", f"Robot: {rps_robot_move}"], 80, NOTE_COLOR, scale=0.65, thickness=2, step=30)
        draw_lines(panel, [rps_result_text], 190, rps_result_color, scale=1.0, thickness=2)
        draw_lines(panel, ["Thumbs up = Start again", "Voice: START", "OK sign = Back to Menu", "Voice: BACK"], 240, OPTION_COLOR, scale=0.65, thickness=2, step=25)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            send_robot_fingers(0)
            rps_start_round()
        elif action == "BACK":
            back_to_previous_menu()


def render_even_odd(panel, camera_view, current_time, all_hand_landmarks, first_hand_landmarks, current_fingers_single, current_fingers_two, current_ok, current_thumbs, h, w):
    # Handles logic and UI updates for the Even/Odd mode
    global even_state, even_selected_side, even_prev_distances, even_locked_features, even_has_predicted
    global even_predicted_user_move, even_robot_move, even_confidence_percent, even_verify_start, even_candidate_actual
    global even_candidate_actual_start, even_actual_user_move

    current_features = None
    if first_hand_landmarks is not None:
        current_features, even_prev_distances = extract_prediction_features(first_hand_landmarks, even_prev_distances)
    else:
        even_prev_distances = None

    if even_state == "CHOOSE_SIDE":
        draw_lines(panel, ["CHOOSE YOUR SIDE"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, ["Show 2 fingers = EVEN", "Show 1 finger = ODD"], 90, OPTION_COLOR, scale=0.75, thickness=2, step=30)
        draw_lines(panel, ["Voice: say EVEN or ODD", "OK sign = Back to Menu"], 170, NOTE_COLOR, scale=0.65, thickness=2, step=25)
        desired = "BACK" if current_ok else ("EVEN" if current_fingers_single == 2 else ("ODD" if current_fingers_single == 1 else None))
        action, _ = update_hold(desired, current_time)
        if action == "BACK":
            back_to_previous_menu()
        elif action in ["EVEN", "ODD"]:
            even_selected_side = action
            even_state = "READY"

    elif even_state == "READY":
        draw_lines(panel, ["EVEN ODD GAME"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Your side: {even_selected_side}", "Thumbs up = Start game"], 90, OPTION_COLOR, scale=0.75, thickness=2, step=30)
        draw_lines(panel, ["OK sign = Back to Menu"], 170, NOTE_COLOR, scale=0.65, thickness=2, step=25)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            even_start_round()
        elif action == "BACK":
            back_to_previous_menu()

    elif even_state == "COUNTDOWN":
        elapsed = current_time - even_countdown_start
        count = int(math.ceil(max(0.0, COUNTDOWN_DURATION - elapsed)))
        draw_lines(panel, ["EVEN ODD GAME"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Side: {even_selected_side}", "Show number now"], 80, OPTION_COLOR, scale=0.75, thickness=2, step=30)
        draw_lines(panel, ["Robot predicts before end"], 150, NOTE_COLOR, scale=0.65, thickness=2)
        if count > 0:
            cv2.putText(camera_view, str(count), (w // 2 - 45, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 4.6, OPTION_COLOR, 8)
        if current_fingers_single != -1:
            cv2.putText(panel, f"Detected: {current_fingers_single}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
        draw_lines(panel, ["OK sign = Back"], 240, NOTE_COLOR, scale=0.65, thickness=2)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            back_to_previous_menu()
            return
        if elapsed >= FEATURE_LOCK_TIME and even_locked_features is None and current_features is not None:
            even_locked_features = list(current_features)
        if elapsed >= COUNTDOWN_DURATION and not even_has_predicted:
            if even_locked_features is not None and np is not None and even_odd_model is not None:
                probs = even_odd_model.predict_proba([even_locked_features])[0]
                even_predicted_user_move = int(np.argmax(probs))
                even_confidence_percent = float(probs[even_predicted_user_move] * 100)
            else:
                even_predicted_user_move = random.randint(0, 5)
            even_robot_move = 1 if ((even_predicted_user_move % 2 == 0 and even_selected_side == "EVEN") or (even_predicted_user_move % 2 != 0 and even_selected_side == "ODD")) else 2
            send_robot_fingers(even_robot_move)
            even_has_predicted = True
            even_state = "VERIFY_ACTUAL"
            even_verify_start = current_time
            even_candidate_actual = None

    elif even_state == "VERIFY_ACTUAL":
        elapsed = current_time - even_verify_start
        draw_lines(panel, ["FINAL CHECK"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Robot played: {even_robot_move}", "Hold final number"], 80, OPTION_COLOR, scale=0.75, thickness=2, step=30)
        draw_lines(panel, [f"Predicted: {even_predicted_user_move} ({even_confidence_percent:.1f}%)"], 150, NOTE_COLOR, scale=0.65, thickness=2)
        if current_fingers_single != -1:
            cv2.putText(panel, f"Detected: {current_fingers_single}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
            if even_candidate_actual != current_fingers_single:
                even_candidate_actual = current_fingers_single
                even_candidate_actual_start = current_time
            else:
                hold = current_time - even_candidate_actual_start
                cv2.putText(panel, f"Stable hold: {hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.65, HOLD_COLOR, 2)
                if hold >= ANSWER_STABLE_SECONDS:
                    even_actual_user_move = current_fingers_single
                    even_finish_result()
        else:
            even_candidate_actual = None
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            back_to_previous_menu()
            return
        if elapsed >= VERIFY_ACTUAL_SECONDS and even_state == "VERIFY_ACTUAL":
            even_actual_user_move = current_fingers_single
            even_finish_result()

    elif even_state == "RESULT":
        total = even_actual_user_move + even_robot_move if even_actual_user_move >= 0 else "?"
        draw_lines(panel, ["RESULT"], 40, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Side: {even_selected_side}", f"Predict: {even_predicted_user_move} ({even_confidence_percent:.1f}%)", f"Actual: {even_actual_user_move}", f"Robot: {even_robot_move}", f"Total: {total}"], 80, NOTE_COLOR, scale=0.65, thickness=2, step=25)
        draw_lines(panel, [even_result_text], 220, even_result_color, scale=1.0, thickness=2)
        draw_lines(panel, ["Thumbs up = Start again", "Voice: START", "OK sign = Back to Menu"], 260, OPTION_COLOR, scale=0.65, thickness=2, step=25)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            send_robot_fingers(0)
            even_start_round()
        elif action == "BACK":
            back_to_previous_menu()


def render_counting(panel, current_time, current_fingers_single, current_ok, current_thumbs, h):
    # Handles logic and UI updates for the Counting learning mode
    global count_state, count_candidate_fingers, count_candidate_start_time, count_stable_finger_answer

    if SIMULATION_MODE:
        cv2.putText(panel, "SIMULATION MODE", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NOTE_COLOR, 2)

    if count_state == "READY":
        draw_lines(panel, ["COUNTING / IMITATION"], 70, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, ["Thumbs up = Start learning", "Voice: say START"], 130, OPTION_COLOR, scale=0.85, thickness=2, step=35)
        draw_lines(panel, ["OK sign = Back to Menu", "Voice: say BACK"], 220, NOTE_COLOR, scale=0.7, thickness=2, step=35)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            count_start_round()
        elif action == "BACK":
            back_to_previous_menu()

    elif count_state == "WAIT_FOR_USER":
        draw_lines(panel, ["COPY THE ROBOT"], 70, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Robot shows: {count_target_number}"], 120, OPTION_COLOR, scale=0.9, thickness=2)
        draw_lines(panel, ["Show same number with fingers", "Say number out loud", "OK sign = Back"], 170, NOTE_COLOR, scale=0.65, thickness=2, step=30)
        current_y = 280
        if current_fingers_single != -1:
            cv2.putText(panel, f"Detected fingers: {current_fingers_single}", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
            current_y += 35
            if count_candidate_fingers != current_fingers_single:
                count_candidate_fingers = current_fingers_single
                count_candidate_start_time = current_time
            else:
                finger_hold = current_time - count_candidate_start_time
                cv2.putText(panel, f"Hold: {finger_hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                current_y += 35
                if finger_hold >= ANSWER_STABLE_SECONDS:
                    count_stable_finger_answer = current_fingers_single
        if count_spoken_number is not None:
            cv2.putText(panel, f"Heard number: {count_spoken_number}", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
            current_y += 35
        elif count_heard_text:
            cv2.putText(panel, f"Heard: {count_heard_text}", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INFO_COLOR, 2)
            current_y += 35
        if count_stable_finger_answer is not None:
            if voice_enabled:
                if count_spoken_number is not None:
                    count_go_to_feedback(count_stable_finger_answer, count_spoken_number)
                else:
                    cv2.putText(panel, "Waiting for voice...", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
            else:
                count_go_to_feedback(count_stable_finger_answer, None)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            back_to_previous_menu()

    elif count_state == "FEEDBACK":
        draw_lines(panel, [count_feedback_text], h // 2 - 20, count_feedback_color, scale=0.8, thickness=2)
        if current_time - count_feedback_start_time >= COUNTING_FEEDBACK_SECONDS:
            count_state = "ROUND_END_MENU"

    elif count_state == "ROUND_END_MENU":
        draw_lines(panel, ["ROUND FINISHED"], 70, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [count_feedback_text], 130, count_feedback_color, scale=0.8, thickness=2)
        draw_lines(panel, ["Thumbs up = Start again", "Voice: say START", "OK sign = Back", "Voice: say BACK"], 190, OPTION_COLOR, scale=0.7, thickness=2, step=35)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            count_start_round()
        elif action == "BACK":
            back_to_previous_menu()


def render_big_small(panel, current_time, all_hand_landmarks, current_fingers_two, current_ok, current_thumbs, h):
    # Handles logic and UI updates for the Greater/Smaller learning mode
    global gs_state, gs_selected_mode, gs_candidate_answer, gs_candidate_start_time, gs_state_start_time
    elapsed = current_time - gs_state_start_time

    if gs_state == "CHOOSE_MODE":
        draw_lines(panel, ["GREATER / SMALLER MODE"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, ["Show 1 finger = Greater", "Show 2 fingers = Smaller"], 100, OPTION_COLOR, scale=0.8, thickness=2, step=30)
        draw_lines(panel, ["Voice: say GREATER / SMALLER", "OK sign = Back to Education Menu", "Voice: BACK"], 180, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        desired = None
        if current_ok:
            desired = "BACK"
        elif current_fingers_two == 1:
            desired = "GREATER"
        elif current_fingers_two == 2:
            desired = "SMALLER"
        action, _ = update_hold(desired, current_time)
        if action == "BACK":
            back_to_previous_menu()
        elif action == "GREATER":
            gs_selected_mode = "GREATER"
            gs_state = "READY"
        elif action == "SMALLER":
            gs_selected_mode = "SMALLER"
            gs_state = "READY"

    elif gs_state == "READY":
        draw_lines(panel, ["GREATER / SMALLER MODE"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Chosen mode: {gs_selected_mode}", "Thumbs up = Start game", "Voice: START"], 100, OPTION_COLOR, scale=0.8, thickness=2, step=30)
        draw_lines(panel, ["OK sign = Back to mode choice", "Voice: BACK"], 200, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            gs_start_new_exercise()
        elif action == "BACK":
            gs_go_to_mode_menu()

    elif gs_state == "SHOW_FIRST":
        draw_lines(panel, ["WATCH THE ROBOT"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"First number: {gs_first_number}"], 110, OPTION_COLOR, scale=0.9, thickness=2)
        draw_lines(panel, [f"Mode: {gs_selected_mode}", "OK sign = Back to mode choice", "Voice: BACK"], 170, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            gs_go_to_mode_menu()
            return
        if elapsed >= ROBOT_SHOW_SECONDS and not voice_guide.is_busy:
            send_robot_fingers(0)
            gs_state = "GAP"
            gs_state_start_time = current_time

    elif gs_state == "GAP":
        draw_lines(panel, ["GET READY FOR NEXT NUMBER"], h // 2, NOTE_COLOR, scale=0.8, thickness=2)
        if elapsed >= ROBOT_GAP_SECONDS:
            send_robot_fingers(gs_second_number)
            gs_state = "SHOW_SECOND"
            gs_state_start_time = current_time

    elif gs_state == "SHOW_SECOND":
        draw_lines(panel, ["WATCH THE ROBOT"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Second number: {gs_second_number}"], 110, OPTION_COLOR, scale=0.9, thickness=2)
        draw_lines(panel, [f"Question: Which is {gs_selected_mode.lower()}?", "OK sign = Back to mode choice", "Voice: BACK"], 170, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            gs_go_to_mode_menu()
            return
        if elapsed >= ROBOT_SHOW_SECONDS and not voice_guide.is_busy:
            send_robot_fingers(0)
            gs_candidate_answer = None
            gs_candidate_start_time = 0.0
            gs_state = "WAIT_FOR_ANSWER"
            gs_state_start_time = current_time

    elif gs_state == "WAIT_FOR_ANSWER":
        draw_lines(panel, ["ANSWER THE QUESTION"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Numbers: {gs_first_number} and {gs_second_number}"], 100, OPTION_COLOR, scale=0.95, thickness=2)
        draw_lines(panel, [f"Show the {gs_selected_mode.lower()} number", "Hold answer for 0.8 seconds", "OK sign = Back to mode choice", "Voice: BACK"], 160, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        cv2.putText(panel, f"Level: {gs_current_level} / {MAX_LEVEL}", (30, h - 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, INFO_COLOR, 2)
        if current_fingers_two != -1:
            cv2.putText(panel, f"Detected fingers: {current_fingers_two}", (30, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, INFO_COLOR, 2)
            if gs_candidate_answer != current_fingers_two:
                gs_candidate_answer = current_fingers_two
                gs_candidate_start_time = current_time
            else:
                hold = current_time - gs_candidate_start_time
                cv2.putText(panel, f"Answer hold: {hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                if hold >= ANSWER_STABLE_SECONDS:
                    gs_check_answer(current_fingers_two)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            gs_go_to_mode_menu()

    elif gs_state == "FEEDBACK":
        draw_lines(panel, [gs_feedback_text], h // 2 - 20, gs_feedback_color, scale=1.0, thickness=3)
        if elapsed >= FEEDBACK_SECONDS:
            gs_state = "ROUND_END_MENU"
            gs_state_start_time = current_time

    elif gs_state == "ROUND_END_MENU":
        draw_lines(panel, ["ROUND FINISHED"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [gs_feedback_text], 100, gs_feedback_color, scale=0.9, thickness=2)
        msg = "Thumbs up = Next exercise" if gs_last_answer_correct else "Thumbs up = Try again"
        draw_lines(panel, [msg, "Voice: START", "OK sign = Back to mode choice", "Voice: BACK"], 160, OPTION_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            if gs_last_answer_correct:
                gs_start_new_exercise()
            else:
                gs_candidate_answer = None
                gs_candidate_start_time = 0.0
                gs_state = "WAIT_FOR_ANSWER"
                gs_state_start_time = current_time
        elif action == "BACK":
            gs_go_to_mode_menu()


def render_math(panel, current_time, all_hand_landmarks, current_fingers_two, current_ok, current_thumbs, h):
    # Handles logic and UI updates for the Math (Addition/Subtraction) mode
    global math_state, math_selected_operation, math_candidate_answer, math_candidate_start_time, math_state_start_time
    elapsed = current_time - math_state_start_time

    if math_state == "CHOOSE_OPERATION":
        draw_lines(panel, ["ADDITION / SUBTRACTION"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, ["Show 1 finger = Addition", "Show 2 fingers = Subtraction"], 100, OPTION_COLOR, scale=0.8, thickness=2, step=30)
        draw_lines(panel, ["Voice: say PLUS / MINUS", "OK sign = Back to Education Menu", "Voice: BACK"], 180, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        desired = None
        if current_ok:
            desired = "BACK"
        elif current_fingers_two == 1:
            desired = "ADD"
        elif current_fingers_two == 2:
            desired = "SUB"
        action, _ = update_hold(desired, current_time)
        if action == "BACK":
            back_to_previous_menu()
        elif action == "ADD":
            math_selected_operation = "ADD"
            math_state = "READY"
        elif action == "SUB":
            math_selected_operation = "SUB"
            math_state = "READY"

    elif math_state == "READY":
        draw_lines(panel, ["ADDITION / SUBTRACTION"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Chosen operation: {math_get_operation_symbol()}", "Thumbs up = Start game", "Voice: START"], 100, OPTION_COLOR, scale=0.8, thickness=2, step=30)
        draw_lines(panel, ["OK sign = Back to operation choice", "Voice: BACK"], 200, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            math_start_new_exercise()
        elif action == "BACK":
            math_go_to_operation_menu()

    elif math_state == "SHOW_FIRST":
        draw_lines(panel, ["WATCH THE ROBOT"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"First number: {math_first_number}"], 110, OPTION_COLOR, scale=0.9, thickness=2)
        draw_lines(panel, [f"Operation: {math_get_operation_symbol()}", "OK sign = Back", "Voice: BACK"], 170, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            math_go_to_operation_menu()
            return
        if elapsed >= ROBOT_SHOW_SECONDS and not voice_guide.is_busy:
            send_robot_fingers(0)
            math_state = "GAP"
            math_state_start_time = current_time

    elif math_state == "GAP":
        draw_lines(panel, ["GET READY FOR NEXT NUMBER"], h // 2, NOTE_COLOR, scale=0.8, thickness=2)
        if elapsed >= ROBOT_GAP_SECONDS:
            send_robot_fingers(math_second_number)
            math_state = "SHOW_SECOND"
            math_state_start_time = current_time

    elif math_state == "SHOW_SECOND":
        draw_lines(panel, ["WATCH THE ROBOT"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"Second number: {math_second_number}"], 110, OPTION_COLOR, scale=0.9, thickness=2)
        draw_lines(panel, [f"Question: {math_first_number} {math_get_operation_symbol()} {math_second_number} = ?", "OK sign = Back", "Voice: BACK"], 170, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            math_go_to_operation_menu()
            return
        if elapsed >= ROBOT_SHOW_SECONDS and not voice_guide.is_busy:
            send_robot_fingers(0)
            math_candidate_answer = None
            math_candidate_start_time = 0.0
            math_state = "WAIT_FOR_ANSWER"
            math_state_start_time = current_time

    elif math_state == "WAIT_FOR_ANSWER":
        draw_lines(panel, ["SOLVE THE EXERCISE"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [f"{math_first_number} {math_get_operation_symbol()} {math_second_number} = ?"], 100, OPTION_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, ["Show answer with hands", "Hold your answer for 0.8s", "OK sign = Back to menu", "Voice: BACK"], 160, NOTE_COLOR, scale=0.7, thickness=2, step=30)
        cv2.putText(panel, f"Level: {math_current_level} / {MAX_LEVEL}", (30, h - 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, INFO_COLOR, 2)
        if current_fingers_two != -1:
            cv2.putText(panel, f"Detected fingers: {current_fingers_two}", (30, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, INFO_COLOR, 2)
            if math_candidate_answer != current_fingers_two:
                math_candidate_answer = current_fingers_two
                math_candidate_start_time = current_time
            else:
                hold = current_time - math_candidate_start_time
                cv2.putText(panel, f"Answer hold: {hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                if hold >= ANSWER_STABLE_SECONDS:
                    math_check_answer(current_fingers_two)
        action, _ = update_hold("BACK" if current_ok else None, current_time)
        if action == "BACK":
            math_go_to_operation_menu()

    elif math_state == "FEEDBACK":
        draw_lines(panel, [math_feedback_text], h // 2 - 20, math_feedback_color, scale=1.0, thickness=3)
        if elapsed >= FEEDBACK_SECONDS:
            math_state = "ROUND_END_MENU"
            math_state_start_time = current_time

    elif math_state == "ROUND_END_MENU":
        draw_lines(panel, ["ROUND FINISHED"], 50, TITLE_COLOR, scale=1.0, thickness=2)
        draw_lines(panel, [math_feedback_text], 100, math_feedback_color, scale=0.9, thickness=2)
        msg = "Thumbs up = Next exercise" if math_last_answer_correct else "Thumbs up = Try again"
        draw_lines(panel, [msg, "Voice: START", "OK sign = Back to menu", "Voice: BACK"], 160, OPTION_COLOR, scale=0.7, thickness=2, step=30)
        action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
        if action == "START":
            if math_last_answer_correct:
                math_start_new_exercise()
            else:
                math_candidate_answer = None
                math_candidate_start_time = 0.0
                math_state = "WAIT_FOR_ANSWER"
                math_state_start_time = current_time
        elif action == "BACK":
            math_go_to_operation_menu()


# =========================================================
# Main
# =========================================================
def main():
    # Initializes all systems and contains the main application loop
    global ser
    init_voice()
    ser = init_serial_connection()
    load_models()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    attempts = 0
    while not cap.isOpened() and attempts < 5:
        time.sleep(0.5)
        cap = cv2.VideoCapture(0)
        attempts += 1
    if not cap.isOpened():
        stop_voice()
        if ser is not None and ser.is_open:
            ser.close()
        raise SystemExit("Camera Error")

    try:
        with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue
                frame = cv2.flip(frame, 1)
                camera_view, panel = create_split_screen(frame)
                h, w, _ = frame.shape

                results = hands.process(cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB))
                all_hand_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else []
                first_hand_landmarks = all_hand_landmarks[0] if all_hand_landmarks else None
                
                # Analyze gestures from the hands detected
                current_fingers_single = count_fingers_single_hand(first_hand_landmarks) if first_hand_landmarks is not None else -1
                current_fingers_two = count_fingers_two_hands(all_hand_landmarks)
                current_ok = any(is_ok_gesture(hand) for hand in all_hand_landmarks) if all_hand_landmarks else False
                current_thumbs = any(is_thumbs_up(hand) for hand in all_hand_landmarks) if all_hand_landmarks else False

                for hand_landmarks in all_hand_landmarks:
                    mp_drawing.draw_landmarks(camera_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                current_time = time.time()
                while voice_queue:
                    handle_voice_command(voice_queue.popleft())

                # Route to the appropriate rendering function based on the current active screen
                if current_screen == MAIN_MENU:
                    render_main_menu(panel, current_time, current_fingers_single, current_ok)
                elif current_screen == GAME_MENU:
                    render_game_menu(panel, current_time, current_fingers_single, current_ok)
                elif current_screen == LEARNING_MENU:
                    render_learning_menu(panel, current_time, current_fingers_single, current_ok)
                elif current_screen == SCREEN_RPS:
                    render_rps(panel, camera_view, current_time, all_hand_landmarks, first_hand_landmarks, current_fingers_single, current_ok, current_thumbs, h, w)
                elif current_screen == SCREEN_EVEN_ODD:
                    render_even_odd(panel, camera_view, current_time, all_hand_landmarks, first_hand_landmarks, current_fingers_single, current_fingers_two, current_ok, current_thumbs, h, w)
                elif current_screen == SCREEN_COUNTING:
                    render_counting(panel, current_time, current_fingers_single, current_ok, current_thumbs, h)
                elif current_screen == SCREEN_BIG_SMALL:
                    render_big_small(panel, current_time, all_hand_landmarks, current_fingers_two, current_ok, current_thumbs, h)
                elif current_screen == SCREEN_MATH:
                    render_math(panel, current_time, all_hand_landmarks, current_fingers_two, current_ok, current_thumbs, h)

                voice_key, voice_text = get_voice_payload()
                voice_guide.announce(voice_key, voice_text)

                if current_screen in [MAIN_MENU, GAME_MENU, LEARNING_MENU]:
                    h_panel, _, _ = panel.shape
                    if hold_action:
                        progress = min(SELECTION_HOLD_SECONDS, current_time - hold_start_time)
                        cv2.putText(panel, f"Hold action: {hold_label(hold_action)}", (35, h_panel - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                        cv2.putText(panel, f"Hold selection: {progress:.1f}s / {SELECTION_HOLD_SECONDS:.1f}s", (35, h_panel - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                    else:
                        cv2.putText(panel, "Hold a gesture for 1.5 seconds to select", (35, h_panel - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NOTE_COLOR, 2)
                    if current_fingers_single != -1:
                        cv2.putText(panel, f"Detected fingers: {current_fingers_single}", (35, h_panel - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, STATUS_COLOR, 2)
                else:
                    draw_hold_status(panel, current_time)

                combined_screen = cv2.hconcat([camera_view, panel])
                combined_screen = cv2.resize(combined_screen, (1280, 520))
                window_name = "Robotic Hand - Unified Interface"
                cv2.imshow(window_name, combined_screen)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stop_voice()
        voice_guide.stop()
        send_robot_fingers(0)
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()