import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train7/weights/best.pt')

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to open camera")
    exit()

print("Press 'q' or ESC to exit detection")

cv2.namedWindow('Fruit Detection')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame")
        break

    # Run detection
    results = model(frame)

    # Draw detection results on the frame
    annotated_frame = results[0].plot()

    # Display the results
    cv2.imshow('Fruit Detection', annotated_frame)

    # Press 'q' key or ESC key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 is ESC
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()