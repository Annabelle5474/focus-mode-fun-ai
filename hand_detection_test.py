import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# set the detection and tracking confidence thresholds
hands = mp_hands.Hands(
    # detection confidence threshold is set to 0.5, which means that the model will only detect hands that it is at least 50% confident are present in the frame.
    min_detection_confidence=0.5,
    # tracking confidence threshold is set to 0.5, which means that the model will only track hands that it is at least 50% confident are present in the frame.
    min_tracking_confidence=0.5
)

# 2. Initialize webcam
cap = cv2.VideoCapture(0)

print("Hand Detection Started! Press Q to quit.")

# 3. Continuously capture webcam frames
while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # 4. Convert BGR to RGB
    # The frame captured from the webcam is in BGR format (Blue, Green, Red), which is the default color format used by OpenCV. 
    # However, MediaPipe requires the input image to be in RGB format (Red, Green, Blue). Therefore, we convert the frame from BGR to RGB using the `cv2.cvtColor` function. 
    
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # 5. Detect hands
    results = hands.process(rgb_frame)

    # 6. Draw hand landmarks
    # If hands are detected in the frame, the code iterates through each detected hand and draws the landmarks 
    if results.multi_hand_landmarks:

        # Iterate through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the hand landmarks on the original frame using the `draw_landmarks` function from the `mp.solutions.drawing_utils` module.
            mp.solutions.drawing_utils.draw_landmarks(
                
                # The original frame is passed as the first argument to the `draw_landmarks` function, which is where the landmarks will be drawn.
                frame,
                
                # The second argument is the `hand_landmarks` object, which contains the coordinates of the detected hand landmarks. 
                # This object is obtained from the `results.multi_hand_landmarks` list, which contains the landmarks for each detected hand in the frame.
                hand_landmarks,
                
                # The third argument is the `mp_hands.HAND_CONNECTIONS` object, which defines the connections between the hand landmarks. This object is used to draw lines between the landmarks to create a visual representation of the hand.
                mp_hands.HAND_CONNECTIONS
            )

    # 7. Display webcam
    cv2.imshow("MediaPipe Hand Detection", frame)

    # 8. Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 9. Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
