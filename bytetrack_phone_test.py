#This project demonstrates how to use the Ultralytics YOLO model with ByteTrack for real-time cell phone tracking using a webcam. The code captures video frames from the webcam, detects cell phones in the frames using the YOLO model, and tracks them across frames using ByteTrack. The detected cell phones are highlighted with bounding boxes, and their tracking IDs and confidence scores are displayed on the video feed. 
import cv2
from ultralytics import YOLO

# ==========================================
# 1. Load your existing YOLO model
# ==========================================

model = YOLO("yolo11n.pt")

# ==========================================
# 2. Start webcam
# ==========================================

cap = cv2.VideoCapture(0)

print("ByteTrack Phone Tracking Started!")
print("Press Q to quit.")

# ==========================================
# 3. Process webcam frames
# ==========================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Unable to read webcam frame.")
        break

    # Run YOLO + ByteTrack
    # You're asking Ultralytics to run YOLO detection and ByteTrack association together
    results = model.track(
        frame,
        # ByteTrack can remember the previous tracking state when processing the next frame
        persist=True, 
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Get the result for the current frame
    result = results[0]

    # ==========================================
    # 4. Examine detected objects
    # ==========================================

    if result.boxes is not None:

        for box in result.boxes:

            # Get detected object's class
            class_id = int(box.cls[0])

            # Only track cell phones
            if model.names[class_id] != "cell phone":
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Get confidence score
            confidence = float(box.conf[0])

            # Get tracking ID
            track_id = None

            # Check if the box has an assigned tracking ID. If it does, retrieve it; otherwise, leave it as None.
            if box.id is not None:
                track_id = int(box.id[0])

            # ==================================
            # 5. Draw phone bounding box
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display tracking ID
            if track_id is not None:
                label = f"Phone ID: {track_id}"
            else:
                label = "Phone ID: Not assigned"

            label += f" | Confidence: {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ==========================================
    # 6. Display webcam
    # ==========================================

    cv2.imshow("ByteTrack Phone Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 7. Release resources
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("ByteTrack stopped.")