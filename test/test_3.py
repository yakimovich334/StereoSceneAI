import cv2
import numpy as np
import os

# Шляхи до папок зі стереозображеннями
left_dir = 'Left'
right_dir = 'Right'
output_dir = 'Output'

# Створюємо папку для результатів, якщо вона не існує
os.makedirs(output_dir, exist_ok=True)

# Налаштування для обчислення диспаратності
min_disp = 0
num_disp = 64  # Має бути кратним 16
block_size = 2  # Зменшили розмір блока для більш детальної карти

# Створення об’єкта StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=4 * 3 * block_size ** 2,
    P2=16 * 3 * block_size ** 2,
    disp12MaxDiff=5,
    uniquenessRatio=5,
    speckleWindowSize=5,
    speckleRange=32
)

# Завантаження моделі YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Функція для виконання YOLO на зображенні
def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

# Функція для аналізу результатів YOLO
def get_object_bboxes(outputs, height, width):
    boxes = []
    confidences = []
    class_ids = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
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

    # Перевіряємо, чи існує відповідне зображення у папці right
    if not os.path.exists(right_img_path):
        print(f"[УВАГА] Відсутнє відповідне праве зображення для: {filename}")
        continue

    # Завантаження зображень у кольоровому форматі
    left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)

    # Отримуємо висоту, ширину та кількість каналів
    height, width, channels = left_img.shape

    if left_img is None or right_img is None:
        print(f"[ПОМИЛКА] Не вдалося завантажити: {filename}")
        continue

    # Виявлення об'єктів на лівому зображенні за допомогою YOLO
    height, width, channels = left_img.shape
    outputs = detect_objects(left_img)
    boxes, confidences, class_ids = get_object_bboxes(outputs, height, width)

    # Обчислення карти диспаратності
    disparity_map = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Нормалізація для візуалізації у чорно-білому форматі
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_gray = np.uint8(disparity_map_normalized)

    # Обчислення глибини для об'єктів
    depths = []
    for box in boxes:
        x, y, w, h = box
        roi = disparity_map[y:y+h, x:x+w]
        avg_depth = np.mean(roi)
        depths.append(avg_depth)
       
    # Збереження результату
    output_path = os.path.join(output_dir, f"disparity_{filename}")
    cv2.imwrite(output_path, disparity_map_gray)

    # Відображення результату (за бажанням)
    cv2.imshow("Ліве зображення", left_img)
    cv2.imshow("Праве зображення", right_img)
    cv2.imshow("Карта диспаратності", disparity_map_gray)
    cv2.waitKey(1000)
    # Перевірка натискання клавіші 'q' для виходу
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        print("Роботу програми перервано користувачем.")
        break

cv2.destroyAllWindows()
print("Обробку завершено! Результати збережено у папці 'output'.")