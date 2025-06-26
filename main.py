from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 1 if external camera
cap.set(3, 1280)
cap.set(4, 720)

# Load pre-trained YOLOv5s model
model = YOLO("yolov5s.pt")  # Make sure this file is in your directory or auto-downloads

classNames = model.names  # COCO class names

prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLO detection
    results = model(img, stream=True)

    new_frame_time = time.time()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Get confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Draw bounding box and label
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1, y1), scale=1, thickness=1)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Real-Time Object Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
