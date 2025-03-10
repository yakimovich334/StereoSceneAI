import cv2
import numpy as np
import os

# Шляхи до папок зі стереозображеннями
left_dir = 'image_2'
right_dir = 'image_3'
output_dir = 'Output'

# Створюємо папку для результатів, якщо вона не існує
os.makedirs(output_dir, exist_ok=True)

# Налаштування для обчислення диспаратності
min_disp = 0
num_disp = 256  # Має бути кратним 16
block_size = 7  # Розмір блока для порівняння

# Створення об’єкта StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=50,
    speckleRange=16
)

# Отримуємо список файлів у папці left
image_files = sorted(os.listdir(left_dir))

# Обробка кожної пари зображень
for filename in image_files:
    left_img_path = os.path.join(left_dir, filename)
    right_img_path = os.path.join(right_dir, filename)

    # Перевіряємо, чи існує відповідне зображення у папці right
    if not os.path.exists(right_img_path):
        print(f"[УВАГА] Відсутнє відповідне праве зображення для: {filename}")
        continue

    # Завантаження зображень
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        print(f"[ПОМИЛКА] Не вдалося завантажити: {filename}")
        continue

    # Обчислення карти диспаратності
    disparity_map = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Нормалізація для візуалізації
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)

    # Збереження результату
    output_path = os.path.join(output_dir, f"disparity_{filename}")
    cv2.imwrite(output_path, disparity_map_normalized)

    # Відображення результату (за бажанням)
    cv2.imshow("Ліве зображення", left_img)
    cv2.imshow("Праве зображення", right_img)
    cv2.imshow("Карта диспаратності", disparity_map_normalized)
    cv2.waitKey(1000)
    # Перевірка натискання клавіші 'q' для виходу
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        print("Роботу програми перервано користувачем.")
        break

cv2.destroyAllWindows()
print("Обробку завершено! Результати збережено у папці 'output'.")
