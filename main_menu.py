import cv2
import sys
import math
import time
import subprocess
from pathlib import Path
from collections import deque
import mediapipe as mp

try:
    import speech_recognition as sr
except Exception:
    sr = None


BASE_DIR = Path(__file__).resolve().parent

SCRIPT_RPS = BASE_DIR / "GameMode" / "game_Rock_Paper_Scissors.py"
SCRIPT_EVEN_ODD = BASE_DIR / "GameMode" / "game_Even_Odd.py"
SCRIPT_COUNTING = BASE_DIR / "LearningMode" / "countingMode.py"
SCRIPT_MATH = BASE_DIR / "LearningMode" / "game_Plus_Minus.py"
SCRIPT_GREATER_SMALLER = BASE_DIR / "LearningMode" / "game_Big_Small.py"

MAIN_MENU = "MAIN_MENU"
GAME_MENU = "GAME_MENU"
LEARNING_MENU = "LEARNING_MENU"
menu_state = MAIN_MENU

SELECTION_HOLD_SECONDS = 1.5
RETURN_COOLDOWN = 1.5

# -------------------------------------------------
# Panel design
# -------------------------------------------------

# White instructions panel
PANEL_BG_COLOR = (255, 255, 255)

TITLE_COLOR = (0, 0, 0)
OPTION_COLOR = (0, 0, 220)
NOTE_COLOR = (60, 60, 60)
HOLD_COLOR = (100, 40, 0)
STATUS_COLOR = (40, 40, 40)
DIVIDER_COLOR = (170, 170, 170)

cap = None
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

voice_queue = deque()
stop_listening = None
voice_enabled = False
hold_action = None
hold_start_time = 0.0
last_action_time = 0.0


def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def count_fingers_from_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return -1

    fingers = []

    # Thumb
    fingers.append(
        1
        if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6
        else 0
    )

    # Index, middle, ring, pinky
    tips_idx = [8, 12, 16, 20]
    mips_idx = [6, 10, 14, 18]

    for tip, mip in zip(tips_idx, mips_idx):
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip])
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip])
        fingers.append(1 if mip_dist > 0 and tip_dist / mip_dist > 1.15 else 0)

    return max(0, min(5, sum(fingers)))


def is_ok_gesture(hand_landmarks):
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

    return (
        thumb_index_dist < hand_scale * 0.35
        and is_open(12, 10)
        and is_open(16, 14)
        and is_open(20, 18)
    )


def create_split_screen(frame):
    """
    Creates two parts:
    1. camera_view - the real camera image
    2. panel - clean white panel for instructions
    """
    camera_view = frame.copy()

    panel = frame.copy()
    panel[:] = PANEL_BG_COLOR

    # vertical divider at the left edge of the panel
    panel_h = panel.shape[0]
    cv2.line(panel, (0, 0), (0, panel_h), DIVIDER_COLOR, 3)

    return camera_view, panel


def draw_lines(frame, lines, start_y, color, scale=0.9, thickness=2, step=50):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (35, start_y + i * step),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )


def voice_callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio, language="en-US").upper()
        voice_queue.append(text)
        print(f"Main menu heard: {text}")
    except Exception:
        pass


def init_voice():
    global stop_listening, voice_enabled

    voice_enabled = False
    stop_listening = None

    if sr is None:
        print("SpeechRecognition package not available in main menu")
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

        stop_listening = r.listen_in_background(
            m,
            voice_callback,
            phrase_time_limit=1.2,
        )

        voice_enabled = True
        print("Main menu voice control active")

    except Exception as e:
        print(f"Main menu voice not available: {e}")


def stop_voice():
    global stop_listening, voice_enabled

    if stop_listening is not None:
        try:
            stop_listening(wait_for_stop=False)
        except Exception:
            pass

    stop_listening = None
    voice_enabled = False


def open_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera Error: could not open camera")
        raise SystemExit(1)

    return camera


def release_camera():
    global cap

    if cap is not None and cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()
    time.sleep(0.4)


def run_mode(script_path):
    global cap, last_action_time, hold_action, hold_start_time

    if not script_path.exists():
        print(f"Missing script: {script_path}")
        time.sleep(1.2)
        return

    print(f"Opening mode: {script_path.name}")

    stop_voice()
    release_camera()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
    )

    print(f"Mode finished with return code: {result.returncode}")

    time.sleep(0.6)

    cap = open_camera()
    init_voice()

    hold_action = None
    hold_start_time = 0.0
    last_action_time = time.time() + RETURN_COOLDOWN


def update_hold(desired_action, current_time):
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

    if elapsed >= SELECTION_HOLD_SECONDS:
        action = hold_action
        hold_action = None
        hold_start_time = 0.0
        return action, elapsed

    return None, elapsed


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


def draw_hold_status(panel, current_time):
    h, _, _ = panel.shape

    if hold_action:
        progress = min(SELECTION_HOLD_SECONDS, current_time - hold_start_time)

        cv2.putText(
            panel,
            f"Hold action: {hold_label(hold_action)}",
            (35, h - 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            HOLD_COLOR,
            2,
        )

        cv2.putText(
            panel,
            f"Hold selection: {progress:.1f}s / {SELECTION_HOLD_SECONDS:.1f}s",
            (35, h - 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            HOLD_COLOR,
            2,
        )

    else:
        cv2.putText(
            panel,
            "Hold a gesture for 1.5 seconds to select",
            (35, h - 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            NOTE_COLOR,
            2,
        )


def draw_status(panel, current_fingers, voice_text):
    h, _, _ = panel.shape

    if current_fingers != -1:
        cv2.putText(
            panel,
            f"Detected fingers: {current_fingers}",
            (35, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            STATUS_COLOR,
            2,
        )

    if voice_text:
        cv2.putText(
            panel,
            f"Voice: {voice_text}",
            (35, h - 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            STATUS_COLOR,
            2,
        )


def draw_main_menu(panel):
    draw_lines(
        panel,
        ["MAIN MENU"],
        70,
        TITLE_COLOR,
        scale=1.2,
        thickness=3,
    )

    draw_lines(
        panel,
        [
            "Show 1 finger = Game Mode",
            "Show 2 fingers = Learning Mode",
        ],
        145,
        OPTION_COLOR,
        scale=0.95,
        thickness=2,
        step=55,
    )

    draw_lines(
        panel,
        [
            "Voice: say GAME or LEARNING",
            "OK sign = Back (in sub menus)",
        ],
        280,
        NOTE_COLOR,
        scale=0.75,
        thickness=2,
        step=40,
    )


def draw_game_menu(panel):
    draw_lines(
        panel,
        ["GAME MODE"],
        70,
        TITLE_COLOR,
        scale=1.2,
        thickness=3,
    )

    draw_lines(
        panel,
        [
            "Show 1 finger = Rock Paper Scissors",
            "Show 2 fingers = Even Odd",
        ],
        145,
        OPTION_COLOR,
        scale=0.85,
        thickness=2,
        step=55,
    )

    draw_lines(
        panel,
        [
            "OK sign = Back to Main Menu",
            "Voice: ROCK / EVEN / BACK",
        ],
        280,
        NOTE_COLOR,
        scale=0.75,
        thickness=2,
        step=40,
    )


def draw_learning_menu(panel):
    draw_lines(
        panel,
        ["LEARNING MODE"],
        70,
        TITLE_COLOR,
        scale=1.2,
        thickness=3,
    )

    draw_lines(
        panel,
        [
            "Show 1 finger = Counting / Imitation",
            "Show 2 fingers = Addition / Subtraction",
            "Show 3 fingers = Greater / Smaller",
        ],
        140,
        OPTION_COLOR,
        scale=0.78,
        thickness=2,
        step=50,
    )

    draw_lines(
        panel,
        [
            "OK sign = Back to Main Menu",
            "Voice: COUNTING / MATH / GREATER / BACK",
        ],
        310,
        NOTE_COLOR,
        scale=0.68,
        thickness=2,
        step=40,
    )


def handle_voice_command(text):
    global menu_state, last_action_time

    if not text:
        return False

    if menu_state == MAIN_MENU:
        if any(word in text for word in ["GAME"]):
            menu_state = GAME_MENU
            last_action_time = time.time()
            return True

        if any(word in text for word in ["EDUCATION", "LEARNING"]):
            menu_state = LEARNING_MENU
            last_action_time = time.time()
            return True

    elif menu_state == GAME_MENU:
        if "BACK" in text:
            menu_state = MAIN_MENU
            last_action_time = time.time()
            return True

        if "ROCK" in text or "SCISSORS" in text:
            run_mode(SCRIPT_RPS)
            return True

        if "EVEN" in text or "ODD" in text:
            run_mode(SCRIPT_EVEN_ODD)
            return True

    elif menu_state == LEARNING_MENU:
        if "BACK" in text:
            menu_state = MAIN_MENU
            last_action_time = time.time()
            return True

        if "COUNT" in text or "IMITATION" in text:
            run_mode(SCRIPT_COUNTING)
            return True

        if (
            "MATH" in text
            or "PLUS" in text
            or "MINUS" in text
            or "ADD" in text
            or "SUB" in text
        ):
            run_mode(SCRIPT_MATH)
            return True

        if "GREATER" in text or "SMALLER" in text or "BIGGER" in text:
            run_mode(SCRIPT_GREATER_SMALLER)
            return True

    return False


cap = open_camera()
init_voice()

print("Main Menu Started")
print("Press q to quit")

try:
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
    ) as hands:

        while True:
            if cap is None or not cap.isOpened():
                cap = open_camera()

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            # Split screen:
            # camera_view = clean camera side
            # panel = instructions side
            camera_view, panel = create_split_screen(frame)

            rgb_frame = cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_fingers = -1
            ok_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks ONLY on camera side
                    mp_drawing.draw_landmarks(
                        camera_view,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    current_fingers = count_fingers_from_landmarks(hand_landmarks)
                    ok_detected = is_ok_gesture(hand_landmarks)

            current_time = time.time()
            latest_voice = None

            while voice_queue:
                latest_voice = voice_queue.popleft()

                if handle_voice_command(latest_voice):
                    break

            if menu_state == MAIN_MENU:
                draw_main_menu(panel)

                desired_action = None

                if current_time > last_action_time:
                    if current_fingers == 1:
                        desired_action = "go_game"
                    elif current_fingers == 2:
                        desired_action = "go_learning"

                action, _ = update_hold(desired_action, current_time)

                if action == "go_game":
                    menu_state = GAME_MENU
                    last_action_time = current_time + 0.2

                elif action == "go_learning":
                    menu_state = LEARNING_MENU
                    last_action_time = current_time + 0.2

            elif menu_state == GAME_MENU:
                draw_game_menu(panel)

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
                    menu_state = MAIN_MENU
                    last_action_time = current_time + 0.2

                elif action == "open_rps":
                    last_action_time = current_time + 0.2
                    run_mode(SCRIPT_RPS)
                    continue

                elif action == "open_evenodd":
                    last_action_time = current_time + 0.2
                    run_mode(SCRIPT_EVEN_ODD)
                    continue

            elif menu_state == LEARNING_MENU:
                draw_learning_menu(panel)

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
                    menu_state = MAIN_MENU
                    last_action_time = current_time + 0.2

                elif action == "open_counting":
                    last_action_time = current_time + 0.2
                    run_mode(SCRIPT_COUNTING)
                    continue

                elif action == "open_math":
                    last_action_time = current_time + 0.2
                    run_mode(SCRIPT_MATH)
                    continue

                elif action == "open_gs":
                    last_action_time = current_time + 0.2
                    run_mode(SCRIPT_GREATER_SMALLER)
                    continue

            # These are drawn only on the instructions panel
            draw_hold_status(panel, current_time)
            draw_status(
                panel,
                current_fingers,
                latest_voice if voice_enabled else "voice unavailable",
            )

            # Combine camera side + instructions side
            combined_screen = cv2.hconcat([camera_view, panel])

            # Optional resize so the window is not too huge.
            # You can change this or delete it if you want full size.
            combined_screen = cv2.resize(combined_screen, (1280, 520))

            cv2.imshow("Robotic Hand - Main Interface", combined_screen)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    stop_voice()
    release_camera()
    print("Main Menu Closed")