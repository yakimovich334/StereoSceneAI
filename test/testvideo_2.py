import cv2
import numpy as np
import os

# Відеофайл або вебкамера (0 - основна камера)
left_video_path = "left_video.mp4"
right_video_path = "right_video.mp4"

# Відкриваємо відео
left_cap = cv2.VideoCapture(left_video_path)
right_cap = cv2.VideoCapture(right_video_path)

# Перевірка, чи відкрито відеофайли
if not left_cap.isOpened() or not right_cap.isOpened():
    print("[ПОМИЛКА] Не вдалося відкрити відеофайли.")
    exit()

# Налаштування для обчислення диспаратності
min_disp = 0
num_disp = 64  # Має бути кратним 16
block_size = 5

# Створення об’єкта StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * block_size ** 2,
    P2=32 * block_size ** 2,
    disp12MaxDiff=5,
    uniquenessRatio=5,
    speckleWindowSize=5,
    speckleRange=32
)

while left_cap.isOpened() and right_cap.isOpened():
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()

    if not ret_left or not ret_right:
        print("[ІНФО] Кінець відео.")
        break

    # Зміна розміру для покращення продуктивності
    left_frame = cv2.resize(left_frame, (640, 480))
    right_frame = cv2.resize(right_frame, (640, 480))

    # Перетворення в сірий формат для обчислення карти глибин
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Обчислення карти диспаратності
    disparity_map = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Нормалізація
    disparity_map_gray = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Відображення
    cv2.imshow("Ліве зображення", left_frame)
    cv2.imshow("Праве зображення", right_frame)
    cv2.imshow("Карта глибин (ч/б)", disparity_map_gray)

    # Натискання 'q' для виходу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закриття відео та вікон
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()