import cv2
import mediapipe as mp
import math
import time
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

mp_hands = mp.solutions.hands

def get_dist(p1, p2): 
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

cap = cv2.VideoCapture(0)
X_data = []
y_data = []

target_moves = [0, 1, 2, 3, 4, 5]
prev_distances = None

with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7) as hands:
    for move in target_moves:
        print(f"\n--- Prepare to show: {move} ---")
        print("You have 30 seconds! Move your hand dynamically, change distances and angles.")
        time.sleep(3)
        
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 שניות של איסוף לכל מספר
            success, frame = cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
                    
                    if hand_scale == 0: continue
                    
                    # 1. מרחקי קצות האצבעות מהשורש (5 מאפיינים)
                    tips = [4, 8, 12, 16, 20]
                    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
                    
                    # 2. חישוב מהירות השינוי מהפריים הקודם (5 מאפיינים)
                    if prev_distances is None:
                        speeds = [0.0] * 5
                    else:
                        speeds = [cur - prv for cur, prv in zip(current_distances, prev_distances)]
                    
                    prev_distances = current_distances
                    
                    # 3. מרחקים בין אצבעות שכנות - פיסוק (4 מאפיינים)
                    inter_finger_dist = []
                    for i in range(len(tips) - 1):
                        d = get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i+1]]) / hand_scale
                        inter_finger_dist.append(d)
                    
                    # איחוד ל-14 מאפיינים
                    features = current_distances + speeds + inter_finger_dist
                    
                    X_data.append(features)
                    y_data.append(move)
            else:
                prev_distances = None
            
            # חישוב והצגת הזמן הנותר על המסך
            time_left = int(30 - (time.time() - start_time))
            cv2.putText(frame, f"Recording move: {move} ({time_left}s)", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Advanced Data Collector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

# אימון המודל המשופר על מאגר הנתונים המורחב
if X_data:
    print(f"\nTraining the model on {len(X_data)} frames...")
    clf = RandomForestClassifier(n_estimators=200, min_samples_split=4, random_state=42)
    clf.fit(X_data, y_data)
    
    with open("hand_predictor.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("✅ Advanced Model trained and saved successfully as 'hand_predictor.pkl'!")
else:
    print("❌ No data collected.")