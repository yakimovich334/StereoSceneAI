import cv2
import numpy as np

# Вибір камер
left_camera_index = int(input("Введіть індекс лівої камери: "))
right_camera_index = int(input("Введіть індекс правої камери: "))

# Відкриваємо камери
left_cap = cv2.VideoCapture(left_camera_index)
right_cap = cv2.VideoCapture(right_camera_index)

# Перевірка, чи вдалося відкрити камери
if not left_cap.isOpened() or not right_cap.isOpened():
    print("[ПОМИЛКА] Не вдалося відкрити камери.")
    exit()

# Налаштування алгоритму StereoSGBM
min_disp = 0
num_disp = 64  # Має бути кратним 16
block_size = 5
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

while True:
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()

    if not ret_left or not ret_right:
        print("[ПОМИЛКА] Не вдалося отримати кадри з камер.")
        break

    # Зміна розміру для покращення продуктивності
    left_frame = cv2.resize(left_frame, (640, 480))
    right_frame = cv2.resize(right_frame, (640, 480))

    # Перетворення в сірий формат
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Обчислення карти диспаратності
    disparity_map = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity_map_gray = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Відображення результатів
    cv2.imshow("Ліве зображення", left_frame)
    cv2.imshow("Праве зображення", right_frame)
    cv2.imshow("Карта глибин (ч/б)", disparity_map_gray)

    # Натискання 'q' для виходу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закриття камер і вікон
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()
