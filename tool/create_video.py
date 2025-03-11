import cv2
import os

def images_to_video(image_folder, output_video, fps=30):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the folder.")
        return
    
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved as {output_video}")

# Використання
image_folder = "Right"  # Змініть на шлях до ваших зображень
output_video = "Right_video.mp4"
images_to_video(image_folder, output_video)