from picamera2 import Picamera2, Preview
import cv2

# Initialize the camera
picam2 = Picamera2()

# Configure the camera
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

# Start the camera
picam2.start()

while True:
    # Capture a frame
    frame = picam2.capture_array()
    
    # Display the frame
    cv2.imshow("Webcam Feed", frame)
    
    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
picam2.stop()
cv2.destroyAllWindows()