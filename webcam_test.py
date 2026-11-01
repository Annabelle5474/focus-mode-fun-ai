import cv2

print("Starting webcam test...")

# Access  webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Check camera connection
if not cap.isOpened():
    print("ERROR: Cannot access webcam!")
    exit()

print("SUCCESS: Webcam connected!")

# Check if the webcam is opened successfully
while True:

    # Read a frame from the webcam
    success, frame = cap.read()

    # Check if the frame was captured successfully
    if not success:
        print("ERROR: Cannot capture video!")
        break
    
    # Display the captured frame in a window
    cv2.imshow(" Webcam Test", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Webcam closed.")
