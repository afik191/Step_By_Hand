import cv2
import sys
import math
import time
import serial
import pickle
import numpy as np
import mediapipe as mp
import speech_recognition as sr

# --- הגדרות משתנים גלובליים ---
game_state = "CHOOSING" 
user_side = "" 
countdown_start = 0
has_predicted_this_round = False

def voice_callback(recognizer, audio):
    global game_state, user_side, countdown_start, has_predicted_this_round, ser
    try:
        # פענוח הדיבור לאנגלית
        command = recognizer.recognize_google(audio, language="en-US").upper()
        print(f"🎙️ Heard voice command: {command}")
        
        if game_state == "CHOOSING":
            # בדיקת וריאציות נפוצות למילה EVEN
            if any(word in command for word in ["EVEN", "EVENT", "EVAN", "IF"]):
                user_side = "EVEN"
                game_state = "COUNTDOWN"
                countdown_start = time.time()
                has_predicted_this_round = False
                print("➡️ State changed to COUNTDOWN (EVEN) via Voice")
            
            # בדיקת וריאציות נפוצות למילה ODD (כולל שיבושים נפוצים של המנוע)
            elif any(word in command for word in ["ODD", "ADD", "OLD", "ALL", "OH", "OUT", "NOT"]):
                user_side = "ODD"
                game_state = "COUNTDOWN"
                countdown_start = time.time()
                has_predicted_this_round = False
                print("➡️ State changed to COUNTDOWN (ODD) via Voice")
                
        elif game_state == "RESULT":
            # בדיקת וריאציות למילה AGAIN
            if any(word in command for word in ["AGAIN", "GAIN", "GAME", "REMATCH", "BEGIN"]):
                ser.write(b'0')  # החזרת היד למצב אפס
                game_state = "CHOOSING"
                print("➡️ State changed to CHOOSING via Voice")
                
    except sr.UnknownValueError:
        pass  
    except sr.RequestError as e:
        print(f"❌ Voice Service Error: {e}")

# --- אתחול זיהוי קולי אופטימלי ומתוקן ---
r = sr.Recognizer()
m = sr.Microphone()

# שיפורי רגישות וסינון רעשים - סדר הפקודות קריטי למניעת AssertionError
r.energy_threshold = 1000          # רגישות גבוהה ומאוזנת לרעשי מנועים
r.dynamic_energy_threshold = False  # סף קבוע כדי למנוע השתקת המיקרופון עקב רעש הסרוו

# עדכון שני הפרמטרים יחד כדי לעמוד בתנאי: pause_threshold >= non_speaking_duration
r.non_speaking_duration = 0.3      # הגדרת משך זמן אי-הדיבור ל-0.3
r.pause_threshold = 0.3            # קיצור זמן ההמתנה בסוף המילה ל-0.3 לתגובה מיידית

print("🎤 Calibrating microphone context...")
with m as source:
    # דגימה קצרה מאוד של רעש רקע כדי ליצור פילטר
    r.adjust_for_ambient_noise(source, duration=1)

# הפעלת ההקשבה ברקע (phrase_time_limit מוגדר ל-1.2 שניות לחיתוך מהיר)
stop_listening = r.listen_in_background(m, voice_callback, phrase_time_limit=1.2)
print("✅ Multi-modal control active (Voice + Gesture)!")

# --- הגדרות MediaPipe ומודל ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

try:
    with open("hand_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Advanced Predictive ML Model Loaded")
except Exception as e:
    print(f"❌ Could not load ML model: {e}. Run train_predictor.py first.")
    sys.exit()

try:
    ser = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
    print("✅ Connected to Arduino")
except Exception as e:
    print(f"❌ Serial Error: {e}")
    sys.exit()

def get_dist(p1, p2): 
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

cap = cv2.VideoCapture(0)
robot_move = 0
predicted_user_move = -1
confidence_percent = 0.0
prev_distances = None

# טיימרים קטנים למניעת מעברי מצב כפולים ומהירים מדי במחוות ידיים (Debounce)
last_state_change_time = 0

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_features = None
        current_finger_count = -1  # משתנה לספירת אצבעות דטרמיניסטית למצבי מעבר
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
                
                if hand_scale > 0:
                    # ---- א) חילוץ 14 המאפיינים עבור מודל ה-Prediction בשלב ה-Countdown ----
                    tips = [4, 8, 12, 16, 20]
                    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
                    
                    if prev_distances is None:
                        speeds = [0.0] * 5
                    else:
                        speeds = [cur - prv for cur, prv in zip(current_distances, prev_distances)]
                    
                    prev_distances = current_distances
                    
                    inter_finger_dist = []
                    for i in range(len(tips) - 1):
                        d = get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i+1]]) / hand_scale
                        inter_finger_dist.append(d)
                    
                    current_features = current_distances + speeds + inter_finger_dist
                    
                    # ---- ב) ספירת אצבעות מהירה (Rule-based) עבור מסכי התפריט (CHOOSING / RESULT) ----
                    fingers = []
                    # אגודל
                    if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    # 4 אצבעות
                    tips_idx, mips_idx = [8, 12, 16, 20], [6, 10, 14, 18]
                    for t, m in zip(tips_idx, mips_idx):
                        if get_dist(wrist, hand_landmarks.landmark[t]) / get_dist(wrist, hand_landmarks.landmark[m]) > 1.15:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    current_finger_count = sum(fingers)
        else:
            prev_distances = None

        # --- לוגיקת תצוגת ומצבי המשחק (משולב קול + מחווה) ---
        current_time = time.time()
        
        if game_state == "CHOOSING":
            cv2.putText(frame, "Say 'EVEN' or Show 2 Fingers", (30, h//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "Say 'ODD' or Show 1 Finger", (30, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # בדיקת מחוות יד (Debounce של שנייה אחת בין מעברי מצבים)
            if current_time - last_state_change_time > 1.0:
                if current_finger_count == 2:
                    user_side = "EVEN"
                    game_state = "COUNTDOWN"
                    countdown_start = current_time
                    has_predicted_this_round = False
                    last_state_change_time = current_time
                    print("➡️ State changed to COUNTDOWN (EVEN) via Gesture")
                elif current_finger_count == 1:
                    user_side = "ODD"
                    game_state = "COUNTDOWN"
                    countdown_start = current_time
                    has_predicted_this_round = False
                    last_state_change_time = current_time
                    print("➡️ State changed to COUNTDOWN (ODD) via Gesture")

        elif game_state == "COUNTDOWN":
            elapsed = current_time - countdown_start
            count = 3 - int(elapsed)
            
            if count > 0:
                cv2.putText(frame, str(count), (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                
                # ביצוע החיזוי בחצי השנייה האחרונה באמצעות 14 מאפייני ה-ML
                if elapsed >= 2.5 and not has_predicted_this_round and current_features is not None:
                    probabilities = model.predict_proba([current_features])[0]
                    predicted_user_move = int(np.argmax(probabilities))
                    confidence_percent = probabilities[predicted_user_move] * 100
                    
                    if user_side == "EVEN":
                        robot_move = 1 if predicted_user_move % 2 == 0 else 2
                    else:
                        robot_move = 2 if predicted_user_move % 2 == 0 else 1
                    
                    ser.write(str(robot_move).encode())
                    has_predicted_this_round = True
                    print(f"🔮 Prediction: User -> {predicted_user_move} ({confidence_percent:.1f}%) -> Robot plays {robot_move}")
            else:
                game_state = "RESULT"

        elif game_state == "RESULT":
            cv2.putText(frame, f"Predicted Move: {predicted_user_move} ({confidence_percent:.1f}%)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Robot Played: {robot_move}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(frame, "ROBOT WINS!", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.putText(frame, "Say 'AGAIN' or Show Fist (0) for Rematch", (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            
            # חזרה למשחק חדש באמצעות מחווה של יד פתוחה (5 אצבעות)
            if current_time - last_state_change_time > 1.0:
                if current_finger_count == 5:
                    ser.write(b'0')  # החזרת היד למצב אפס (סגורה)
                    game_state = "CHOOSING"
                    last_state_change_time = current_time
                    print("➡️ State changed to CHOOSING via Gesture")

        cv2.imshow('Predictive Zug O Pered - Multi-Modal Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

stop_listening(wait_for_stop=False)
cap.release()
cv2.destroyAllWindows()
ser.close()