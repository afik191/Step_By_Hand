import cv2
import mediapipe as mp
import sys
import time
import math
import serial
from collections import Counter

# הגדרה מפורשת של ה-solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# בדיקה אם זה נטען תקין
if not hasattr(mp, 'solutions'):
    print("❌ MediaPipe error: 'solutions' not found. This is usually caused by a local file named 'mediapipe.py'!")
    sys.exit()

# כעת בקוד, במקום mp.solutions.hands, השתמש פשוט ב-mp_hands:
# with mp_hands.Hands(...) as hands:

# חיבור לארדואינו
try:
    ser = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
    print("✅ Connected to Arduino")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit()

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

cap = cv2.VideoCapture(0)
history = []
game_state = "CHOOSING" 
user_side = "" 
countdown_start = 0
robot_move = 0
final_user_move = 0

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_fingers = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                fingers = []

                # אגודל
                thumb_tip = hand_landmarks.landmark[4]
                index_mcp = hand_landmarks.landmark[5]
                hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
                if get_dist(thumb_tip, index_mcp) > hand_scale * 0.6: fingers.append(1)
                else: fingers.append(0)

                # 4 אצבעות
                tips, mips = [8, 12, 16, 20], [6, 10, 14, 18]
                for t_idx, m_idx in zip(tips, mips):
                    if get_dist(wrist, hand_landmarks.landmark[t_idx]) / get_dist(wrist, hand_landmarks.landmark[m_idx]) > 1.15:
                        fingers.append(1)
                    else: fingers.append(0)
                
                current_fingers = sum(fingers)

        history.append(current_fingers)
        if len(history) > 5: history.pop(0)
        stable_count = Counter(history).most_common(1)[0][0]

        # תצוגת קלט חי
        cv2.rectangle(frame, (w - 220, h - 80), (w - 20, h - 20), (50, 50, 50), -1)
        cv2.putText(frame, f"Hand: {stable_count}", (w - 200, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if game_state == "CHOOSING":
            cv2.putText(frame, "Press 'E' for EVEN, 'O' for ODD", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                user_side, game_state, countdown_start = "EVEN", "COUNTDOWN", time.time()
            elif key == ord('o'):
                user_side, game_state, countdown_start = "ODD", "COUNTDOWN", time.time()

        elif game_state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            count = 3 - int(elapsed)
            if count > 0:
                cv2.putText(frame, str(count), (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
            else:
                # רגע ההחלטה - הרובוט "מרמה" כדי לנצח
                final_user_move = stable_count
                
                if user_side == "EVEN":
                    # המשתמש בחר זוג (הרובוט הוא פרד) -> הרובוט צריך שהסכום יהיה אי-זוגי
                    robot_move = 1 if final_user_move % 2 == 0 else 2
                else:
                    # המשתמש בחר פרד (הרובוט הוא זוג) -> הרובוט צריך שהסכום יהיה זוגי
                    robot_move = 2 if final_user_move % 2 == 0 else 1
                
                # שליחת המהלך לארדואינו
                ser.write(str(robot_move).encode())
                game_state = "RESULT"

        elif game_state == "RESULT":
            cv2.putText(frame, f"Robot Played: {robot_move}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(frame, "ROBOT WINS!", (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            if (cv2.waitKey(1) & 0xFF == ord('r')):
                ser.write(b'0') # החזרת היד למצב סגור
                game_state = "CHOOSING"

        cv2.imshow('ZUG O PERED - Robot Always Wins', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
ser.close()