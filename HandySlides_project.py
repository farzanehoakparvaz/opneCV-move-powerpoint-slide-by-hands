# HandySlides project

import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("web cam does not work")
    exit()

def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]      
    thumb_base = hand_landmarks.landmark[2]    
    index_tip = hand_landmarks.landmark[8]   
    index_base = hand_landmarks.landmark[5]    
    middle_tip = hand_landmarks.landmark[12]   
    middle_base = hand_landmarks.landmark[9]    

    thumb_dist = ((thumb_tip.x - thumb_base.x) ** 2 + (thumb_tip.y - thumb_base.y) ** 2) ** 0.5
    index_dist = ((index_tip.x - index_base.x) ** 2 + (index_tip.y - index_base.y) ** 2) ** 0.5
    middle_dist = ((middle_tip.x - middle_base.x) ** 2 + (middle_tip.y - middle_base.y) ** 2) ** 0.5

    print(f"thumb distance : {thumb_dist:.3f}, index : {index_dist:.3f}, middle :   {middle_dist:.3f}")

    return thumb_dist < 0.2 and index_dist < 0.2 and middle_dist < 0.2


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label 
            print(f"hand detection : {hand_label}")

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_fist(hand_landmarks):
                print(f"{hand_label} hand is fist")
                if hand_label == "Right":
                    pyautogui.press("up")  
                    print("up")
                    time.sleep(0.4)
                elif hand_label == "Left":
                    pyautogui.press("down")  
                    print("down")
                    time.sleep(0.4)
            else:
                print(f"{hand_label} is not fist!")

    else:
        print("hands not detection")

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()