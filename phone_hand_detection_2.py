import cv2
import mediapipe as mp
from ultralytics import YOLO


# ==========================================
# 1. Initialize YOLO
# ==========================================

def initialize_yolo():

    model = YOLO("yolo11n.pt")

    return model


# ==========================================
# 2. Initialize MediaPipe Hands
# ==========================================

def initialize_hands():

    mp_hands = mp.solutions.hands

    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return hands, mp_hands, mp_drawing


# ==========================================
# 3. Initialize Webcam
# ==========================================

def initialize_webcam():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Cannot open webcam!")

        return None

    return cap


# ==========================================
# 4. Check Hand Near Phone
# ==========================================

def is_hand_near_phone(hand_landmarks, phone_box, width, height):

    # Get phone bounding box coordinates
    px1, py1, px2, py2 = phone_box

    # Check all 21 hand landmarks
    for landmark in hand_landmarks.landmark:

        # Convert normalized coordinates to pixel coordinates
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        # Check whether landmark is inside phone bounding box
        if (
            px1 <= x <= px2
            and py1 <= y <= py2
        ):

            return True

    return False


# ==========================================
# 5. Detect Phone Using YOLO
# ==========================================

def detect_phones(model, frame):

    results = model(frame, verbose=False)

    phone_boxes = []

    for result in results:

        for box in result.boxes:

            # Get detected object class
            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            # Check whether object is a phone
            if class_name == "cell phone":

                # Get phone bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                phone_box = (x1, y1, x2, y2)

                phone_boxes.append(phone_box)

                # Draw phone bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                # Display phone label
                cv2.putText(
                    frame,
                    "PHONE",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

    return phone_boxes


# ==========================================
# 6. Detect Hands Using MediaPipe
# ==========================================

def detect_hands(frame, hands, mp_hands, mp_drawing):

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process webcam frame
    hand_results = hands.process(rgb_frame)

    # Store detected hands
    detected_hands = []

    # Check whether a hand is detected
    if hand_results.multi_hand_landmarks:

        for hand_landmarks in hand_results.multi_hand_landmarks:

            # Store hand landmarks
            detected_hands.append(hand_landmarks)

            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return detected_hands


# ==========================================
# 7. Check Hand Near Phone
# ==========================================

def check_hand_near_phone(
    detected_hands,
    phone_boxes,
    width,
    height
):

    hand_near_phone = False

    # Compare every detected hand with every phone
    for hand_landmarks in detected_hands:

        for phone_box in phone_boxes:

            if is_hand_near_phone(
                hand_landmarks,
                phone_box,
                width,
                height
            ):

                hand_near_phone = True

                break

        if hand_near_phone:

            break

    return hand_near_phone


# ==========================================
# 8. Display Detection Messages
# ==========================================

def display_detection_messages(
    frame,
    detected_hands,
    phone_boxes,
    hand_near_phone
):

    # Display hand detection message
    if detected_hands:

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display phone detection message
    if phone_boxes:

        cv2.putText(
            frame,
            "PHONE DETECTED",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # Display hand near phone message
    if hand_near_phone:

        cv2.putText(
            frame,
            "HAND NEAR PHONE",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


# ==========================================
# 9. Main Webcam Loop
# ==========================================

def run_detection(model, hands, mp_hands, mp_drawing, cap):

    while cap.isOpened():

        # Read webcam frame
        success, frame = cap.read()

        if not success:

            print("ERROR: Cannot read webcam frame!")

            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # Detect phone
        phone_boxes = detect_phones(
            model,
            frame
        )

        # Detect hands
        detected_hands = detect_hands(
            frame,
            hands,
            mp_hands,
            mp_drawing
        )

        # Check whether hand is near phone
        hand_near_phone = check_hand_near_phone(
            detected_hands,
            phone_boxes,
            width,
            height
        )

        # Display detection messages
        display_detection_messages(
            frame,
            detected_hands,
            phone_boxes,
            hand_near_phone
        )

        # Display webcam
        cv2.imshow(
            "Focus Mode AI - Phone and Hand Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# 10. Main Function
# ==========================================

def main():

    # Initialize YOLO
    model = initialize_yolo()

    # Initialize MediaPipe
    hands, mp_hands, mp_drawing = initialize_hands()

    # Initialize webcam
    cap = initialize_webcam()

    # Check webcam
    if cap is None:

        hands.close()

        return

    print("Phone and Hand Detection Started!")
    print("Press Q to quit.")

    try:

        # Start detection
        run_detection(
            model,
            hands,
            mp_hands,
            mp_drawing,
            cap
        )

    finally:

        # Release resources
        cap.release()

        cv2.destroyAllWindows()

        hands.close()

        print("Detection stopped.")


# ==========================================
# 11. Run Focus Mode AI
# ==========================================

if __name__ == "__main__":

    main()
