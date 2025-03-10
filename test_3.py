import cv2
import numpy as np
import os
from ultralytics import YOLO
#from PSMNet.models import StackHourglass  # Імпортуйте модель PSMNet (залежить від структури репозиторію)
#from PSMNet.utils import load_state_dict

# Шляхи до папок зі стереозображеннями
left_dir = 'image_2'
right_dir = 'image_3'
output_dir = 'Output'

# Створюємо папку для результатів, якщо вона не існує
os.makedirs(output_dir, exist_ok=True)

# Налаштування для обчислення диспаратності
min_disp = 0
num_disp = 128  # Збільшено для більшої деталізації
block_size = 2   # Збільшено для стабільності та деталей

# Створення об’єкта StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=16 * 3 * block_size ** 2,  # Збільшено для кращої стійкості
    P2=64 * 3 * block_size ** 2,  # Збільшено для кращої стійкості
    disp12MaxDiff=10,
    uniquenessRatio=10,  # Збільшено для зменшення помилок
    speckleWindowSize=100,  # Зменшено для збереження деталей
    speckleRange=32       # Зменшено для меншого шуму
)

# Завантаження моделі YOLOv8
model = YOLO("yolov8n.pt")

# Функція для виконання YOLOv8 на зображенні
def detect_objects(image):
    results = model(image)
    return results

# Функція для аналізу результатів YOLOv8
def get_object_bboxes(results, height, width):
    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            if confidence > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Отримуємо список файлів у папці left
image_files = sorted(os.listdir(left_dir))

# Обробка кожної пари зображень
for filename in image_files:
    left_img_path = os.path.join(left_dir, filename)
    right_img_path = os.path.join(right_dir, filename)

    if not os.path.exists(right_img_path):
        print(f"[УВАГА] Відсутнє відповідне праве зображення для: {filename}")
        continue

    # Завантаження зображень у кольоровому форматі
    left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)

    if left_img is None or right_img is None:
        print(f"[ПОМИЛКА] Не вдалося завантажити: {filename}")
        continue

    # Попередня обробка: фільтрація шуму та підвищення контрасту
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    left_gray = cv2.GaussianBlur(left_gray, (5, 5), 0)
    right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)
    left_gray = cv2.bilateralFilter(left_gray, 9, 75, 75)
    right_gray = cv2.bilateralFilter(right_gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_gray = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)

    # Отримуємо висоту, ширину
    height, width = left_gray.shape

    # Виявлення об'єктів на лівому зображенні за допомогою YOLOv8
    results = detect_objects(left_img)
    boxes, confidences, class_ids = get_object_bboxes(results, height, width)

    # Обчислення карти диспаратності
    disparity_map = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Нормалізація для візуалізації
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)

    # Кольорова палітра (радужний градієнт)
    def custom_colormap(disparity_map):
        colormap = np.zeros((disparity_map.shape[0], disparity_map.shape[1], 3), dtype=np.uint8)
        norm_disparity = disparity_map.astype(float) / 255.0
        
        for i in range(disparity_map.shape[0]):
            for j in range(disparity_map.shape[1]):
                value = norm_disparity[i, j]
                hue = 240 - (value * 240)  # Blue to red via spectrum
                saturation = 1.0
                value_hsv = 1.0
                
                hsv = np.array([hue / 2, saturation * 255, value_hsv * 255], dtype=np.uint8)
                rgb = cv2.cvtColor(np.array([[hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
                colormap[i, j] = rgb
        
        return colormap

    # Застосування кольорової палітри
    disparity_map_colored = custom_colormap(disparity_map_normalized)

    # Обчислення глибини для об'єктів
    depths = []
    for box in boxes:
        x, y, w, h = box
        roi = disparity_map[y:y+h, x:x+w]
        avg_depth = np.mean(roi) if roi.size > 0 else 0
        depths.append(avg_depth)
       
    # Збереження результату
    output_path = os.path.join(output_dir, f"disparity_{filename}")
    cv2.imwrite(output_path, disparity_map_colored)

    # Відображення результату
    cv2.imshow("Ліве зображення", left_img)
    cv2.imshow("Праве зображення", right_img)
    cv2.imshow("Карта диспаратності з кастомною палітрою", disparity_map_colored)
    cv2.waitKey(1000)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Роботу програми перервано користувачем.")
        break

cv2.destroyAllWindows()
print("Обробку завершено! Результати збережено у папці 'output'.")