import cv2

def find_cameras(max_tested=10):
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

cameras = find_cameras()
if cameras:
    print(f"Знайдені камери з індексами: {cameras}")
else:
    print("Камери не знайдено.")
