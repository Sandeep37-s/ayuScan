import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ No camera available")
else:
    print("✅ Camera is now working!")
    cap.release()
