import cv2
# import winsound
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Initialize the webcam
print("--- Focus Project ---")
input("Press Enter to start your Focus Session...")  # Manual trigger

# Webcam only opens AFTER you press Enter
cap = cv2.VideoCapture(0)


# Check if the webcam is opened successfully
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    
    exit()

print("SUCCESS: Webcam opened!")

print("Webcam active. Press 'q' in the video window to stop.")


# Detect if a camera is available
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

# Run the YOLO model on the frame
  results = model(frame, verbose=False)
  phone_detected = False

# Check if a phone is detected in the frame
  for r in results:
    for c in r.boxes.cls:
      if model.names[int(c)] == "cell phone":
        phone_detected = True
        break

# If a phone is detected, play a beep sound and display a warning message on the video frame
  if phone_detected:
      
    # Play a beep sound
    # winsound.Beep(2500, 500)
    
    # Display a warning message on the video frame
    cv2.putText(
        frame,
        "WARNING: PUT THE PHONE AWAY!",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

# Display the video frame with any warnings
  cv2.imshow("Focus Project", frame)

# Check for 'q' key press to exit the loop
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Release the webcam and close all OpenCV windows
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()