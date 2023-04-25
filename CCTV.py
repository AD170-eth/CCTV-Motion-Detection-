import cv2
import numpy as np
import os

# Replace with your DVR's IP address
dvr_ip = "192.168.1.108"

# Replace with your DVR login credentials
username = "admin"
password = "abcd1234"

# Set up the OpenCV video capture objects for all channels
urls = [
    f"rtsp://{username}:{password}@{dvr_ip}/cam/realmonitor?channel={i}&subtype=0" for i in range(1, 9)]
caps = [cv2.VideoCapture(url) for url in urls]

# Create a background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Create directory to store snapshots
snapshot_dir = 'snapshots'
os.makedirs(snapshot_dir, exist_ok=True)

while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction to the frame
        fgmask = fgbg.apply(frame)

        # Apply image erosion and dilation to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a rectangle around each contour with area greater than 500
        for contour in contours:
            if cv2.contourArea(contour) > 20000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Save snapshot to directory
                snapshot_path = os.path.join(
                    snapshot_dir, f"snapshot_{i+1}.jpg")
                cv2.imwrite(snapshot_path, frame)

        # Display the current channel in a separate window
        window_name = f"Channel {i+1}"
        cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture objects and close all windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
