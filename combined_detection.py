import cv2
import mediapipe as mp
from ultralytics import YOLO


# ==================================================
# 1. Initialize YOLO
# ==================================================

# Load the YOLO model.
# yolo11n.pt is a pretrained model that can detect objects,
# including the "cell phone" class.
model = YOLO("yolo11n.pt")


# ==================================================
# 2. Initialize MediaPipe Hands
# ==================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==================================================
# 3. Initialize webcam
# ==================================================

print("--- Focus Mode AI ---")
input("Press Enter to start your focus session...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam!")

    hands.close()
    exit()

print("SUCCESS: Webcam opened!")
print("Press Q to quit.")


# ==================================================
# 4. Continuously capture webcam frames
# ==================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("ERROR: Cannot read webcam frame.")
        break


    # ==================================================
    # 5. Run YOLO phone detection
    # ==================================================

    results = model(frame, verbose=False)

    phone_detected = False

    for result in results:

        # Each detected object has a bounding box.
        for box in result.boxes:

            # Get the detected object's class ID.
            class_id = int(box.cls[0])

            # Get the class name from YOLO.
            class_name = model.names[class_id]

            # Get the detection confidence.
            confidence = float(box.conf[0])

            # Check whether the object is a cell phone.
            if class_name == "cell phone":

                phone_detected = True

                # Get the bounding box coordinates.
                # The `box.xyxy` attribute contains the coordinates of the bounding box in the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. The `map(int, ...)` function converts these coordinates to integers for drawing purposes.
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Draw the phone bounding box.
                cv2.rectangle(
                    # The `cv2.rectangle` function is used to draw a rectangle on the image. The first argument is the image (frame) on which to draw, the second and third arguments are the top-left and bottom-right corners of the rectangle, respectively, and the fourth argument is the color of the rectangle in BGR format (in this case, green). The last argument specifies the thickness of the rectangle's border.
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display the phone label.
                label = f"Phone: {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


    # ==================================================
    # 6. Run MediaPipe hand detection
    # ==================================================

    # OpenCV webcam frames use BGR.
    # MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect hands in the RGB frame.
    hand_results = hands.process(rgb_frame)


    # ==================================================
    # 7. Draw MediaPipe hand landmarks
    # ==================================================

    hand_detected = False

    if hand_results.multi_hand_landmarks:

        hand_detected = True

        for hand_landmarks in hand_results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # ==================================================
    # 8. Display detection status
    # ==================================================

    if phone_detected:

        cv2.putText(
            frame,
            "PHONE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO PHONE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    if hand_detected:

        cv2.putText(
            frame,
            "HAND DETECTED",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # ==================================================
    # 9. Display webcam frame
    # ==================================================

    cv2.imshow(
        "Focus Mode AI - Phone and Hand Detection",
        frame
    )


    # ==================================================
    # 10. Press Q to quit
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==================================================
# 11. Release resources
# ==================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Focus session ended.")
