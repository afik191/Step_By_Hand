import cv2
import sys
import math
import time
import serial
import pickle
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# טעינת מודל החיזוי המשודרג (14 מאפיינים)
try:
    with open("hand_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Advanced Predictive ML Model Loaded")
except Exception as e:
    print(f"❌ Could not load ML model: {e}. Run train_predictor.py first.")
    sys.exit()

# חיבור לארדואינו
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
game_state = "CHOOSING" 
user_side = "" 
countdown_start = 0
robot_move = 0
predicted_user_move = -1
confidence_percent = 0.0  # משתנה חדש לשמירת אחוזי הביטחון
has_predicted_this_round = False

# משתנה לשמירת המרחקים מהפריים הקודם (עבור חישוב המהירות)
prev_distances = None

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_features = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
                
                if hand_scale > 0:
                    # 1. חילוץ 5 מרחקים בסיסיים מהשורש
                    tips = [4, 8, 12, 16, 20]
                    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
                    
                    # 2. חישוב 5 מהירויות שינוי לעומת פריים קודם
                    if prev_distances is None:
                        speeds = [0.0] * 5
                    else:
                        speeds = [cur - prv for cur, prv in zip(current_distances, prev_distances)]
                    
                    prev_distances = current_distances
                    
                    # 3. חישוב 4 מרחקי ביניים (פיסוק אצבעות)
                    inter_finger_dist = []
                    for i in range(len(tips) - 1):
                        d = get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i+1]]) / hand_scale
                        inter_finger_dist.append(d)
                    
                    # איחוד ל-14 מאפיינים
                    current_features = current_distances + speeds + inter_finger_dist
        else:
            prev_distances = None

        # --- לוגיקת המשחק ---
        if game_state == "CHOOSING":
            cv2.putText(frame, "E: EVEN | O: ODD", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'): 
                user_side, game_state, countdown_start, has_predicted_this_round = "EVEN", "COUNTDOWN", time.time(), False
            elif key == ord('o'): 
                user_side, game_state, countdown_start, has_predicted_this_round = "ODD", "COUNTDOWN", time.time(), False

        elif game_state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            count = 3 - int(elapsed)
            
            if count > 0:
                cv2.putText(frame, str(count), (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                
                # ביצוע החיזוי בחצי השנייה האחרונה
                if elapsed >= 2.5 and not has_predicted_this_round and current_features is not None:
                    # מפיקים את מערך ההסתברויות לכל המחלקות (0-5)
                    probabilities = model.predict_proba([current_features])[0]
                    
                    # הניחוש הוא האינדקס עם ההסתברות הגבוהה ביותר
                    predicted_user_move = int(np.argmax(probabilities))
                    
                    # שליפת אחוז הביטחון מתוך המערך
                    confidence_percent = probabilities[predicted_user_move] * 100
                    
                    # קבלת החלטת נגד מבוססת חיזוי
                    if user_side == "EVEN":
                        robot_move = 1 if predicted_user_move % 2 == 0 else 2
                    else:
                        robot_move = 2 if predicted_user_move % 2 == 0 else 1
                    
                    # שליחה מוקדמת לארדואינו
                    ser.write(str(robot_move).encode())
                    has_predicted_this_round = True
                    print(f"🔮 ML Prediction: User -> {predicted_user_move} ({confidence_percent:.1f}%) -> Robot plays {robot_move}")
            else:
                game_state = "RESULT"

        elif game_state == "RESULT":
            # הצגת הניחוש ואחוזי הביטחון על גבי המסך
            cv2.putText(frame, f"Predicted Move: {predicted_user_move} ({confidence_percent:.1f}%)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Robot Played: {robot_move}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(frame, "ROBOT WINS (BY PREDICTION)!", (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4)
            
            if cv2.waitKey(1) & 0xFF == ord('r'):
                ser.write(b'0') 
                game_state = "CHOOSING"

        cv2.imshow('Predictive Zug O Pered', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
ser.close()