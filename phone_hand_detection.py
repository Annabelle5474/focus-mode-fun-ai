import cv2
import mediapipe as mp
from ultralytics import YOLO


# ==========================================
# 1. Load YOLO model
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# 2. Initialize MediaPipe Hands
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 3. Initialize webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Cannot open webcam!")

    hands.close()

    exit()


print("Phone and Hand Detection Started!")
print("Press Q to quit.")


# ==========================================
# 4. Function to check hand landmarks
#    inside phone bounding box
# ==========================================

def is_hand_near_phone(hand_landmarks, phone_box, width, height):

    # Get phone bounding box coordinates
    px1, py1, px2, py2 = phone_box

    # Check all 21 hand landmarks
    for landmark in hand_landmarks.landmark:

        # Convert normalized coordinates to pixel coordinates
        # The `landmark.x` and `landmark.y` values are normalized coordinates in the range [0, 1], representing the relative position of the landmark within the image. 
        # To convert these normalized coordinates to pixel coordinates, we multiply them by the width and height of the frame, respectively. 
        # This gives us the actual pixel positions of the landmarks in the image.
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        # Check whether landmark is inside phone bounding box
        if (
            px1 <= x <= px2
            and py1 <= y <= py2
        ):

            return True

    # No hand landmark overlaps with phone bounding box
    return False


# ==========================================
# 5. Main webcam loop
# ==========================================

while cap.isOpened():

    # Read webcam frame
    success, frame = cap.read()

    if not success:

        print("ERROR: Cannot read webcam frame!")

        break


    # Get frame dimensions
    height, width, _ = frame.shape


    # ==========================================
    # 6. Detect phone using YOLO
    # ==========================================

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


    # ==========================================
    # 7. Detect hand using MediaPipe
    # ==========================================

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


            # Draw only hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # ==========================================
    # 8. Check whether hand landmarks
    #    overlap with phone bounding box
    # ==========================================

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


    # ==========================================
    # 9. Display detection messages
    # ==========================================

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
    # 10. Display webcam
    # ==========================================

    cv2.imshow(
        "Focus Mode AI - Phone and Hand Detection",
        frame
    )


    # ==========================================
    # 11. Press Q to quit
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# 12. Release resources
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Detection stopped.")
