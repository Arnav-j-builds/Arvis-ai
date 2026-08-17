import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Get primary monitor screen dimensions
screen_w, screen_h = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Configuration Variables
frame_reduction = 100       # Boundary padding for easier screen edge reaching
smoothing_factor = 5        # Higher value = smoother but slower cursor movement
prev_x, prev_y = 0, 0       # Stores previous cursor coordinates
current_x, current_y = 0, 0 # Stores current cursor coordinates
click_cooldown = 0.3        # Prevents accidental double clicks
last_click_time = 0

# Disable PyAutoGUI fail-safe to prevent accidental script crashes at screen corners
pyautogui.FAILSAFE = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Flip the frame horizontally for a natural mirror-view mapping
    frame = cv2.flip(frame, 1)
    f_height, f_width, _ = frame.shape
    
    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for Index Tip (8), Middle Tip (12), and Thumb Tip (4)
            landmarks = hand_landmarks.landmark
            
            # Map index finger coordinate to webcam frame pixel dimension
            index_tip = landmarks[8]
            ix, iy = int(index_tip.x * f_width), int(index_tip.y * f_height)
            
            # Draw tracking anchor points on the screen feed
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Coordinate translation from camera box dimensions to monitor pixel boundaries
            # Restricts movement to an inner bounding box for edge-to-edge reach
            mapped_x = np.interp(ix, (frame_reduction, f_width - frame_reduction), (0, screen_w))
            mapped_y = np.interp(iy, (frame_reduction, f_height - frame_reduction), (0, screen_h))
            
            # Apply Linear Interpolation (LERP) Smoothing Formula
            current_x = prev_x + (mapped_x - prev_x) / smoothing_factor
            current_y = prev_y + (mapped_y - prev_y) / smoothing_factor
            
            # Execute mouse cursor movement action
            pyautogui.moveTo(current_x, current_y)
            prev_x, prev_y = current_x, current_y
            
            # Left Click Detection Logic: Distance between Index Tip (8) and Middle Tip (12)
            middle_tip = landmarks[12]
            mx, my = int(middle_tip.x * f_width), int(middle_tip.y * f_height)
            click_distance = np.hypot(mx - ix, my - iy)
            
            # Perform click gesture if fingers touch and cooldown period has expired
            if click_distance < 25:
                if time.time() - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = time.time()
                    # Visual feedback indicator (Green circle)
                    cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                    
    # Render operational frame preview
    cv2.imshow("Smooth AI Virtual Mouse", frame)
    
    # Terminate script execution when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
