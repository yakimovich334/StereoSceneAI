# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import psycopg2
from psycopg2 import sql
import cv2
import numpy as np
import os
from datetime import datetime

# Налаштування підключення до бази даних
DB_CONFIG = {
    "dbname": "StereoAI",
    "user": "Admin",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}

def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Помилка підключення до БД: {e}")
        return None

def close_connection(conn):
    if conn:
        conn.close()
        print("Підключення до БД закрито.")

class SettingsInterface:
    def __init__(self, root, main_interface):
        self.root = root
        self.main_interface = main_interface
        self.root.title("Параметри")
        self.root.geometry("300x400")
        
        try:
            icon = tk.PhotoImage(file=r"C:\Users\nazya\Desktop\StereoSceneAI\icon.png")
            self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Помилка завантаження іконки: {e}")
            
        ttk.Label(root, text="Параметри", anchor="center").pack(pady=10)

        block_size = self.main_interface.block_size if self.main_interface.block_size > 0 else 2
        
        try:
            P1_factor = self.main_interface.stereo.getP1() // (3 * block_size ** 2)
            P2_factor = self.main_interface.stereo.getP2() // (3 * block_size ** 2)
        except (ZeroDivisionError, AttributeError):
            P1_factor = 4
            P2_factor = 16
            print("Warning: Could not calculate P1_factor or P2_factor, using defaults")

        params = [
            ("min_disp", tk.StringVar(value=str(self.main_interface.min_disp))),
            ("num_disp", tk.StringVar(value=str(self.main_interface.num_disp))),
            ("block_size", tk.StringVar(value=str(block_size))),
            ("P1_factor", tk.StringVar(value=str(P1_factor))),
            ("P2_factor", tk.StringVar(value=str(P2_factor))),
            ("disp12MaxDiff", tk.StringVar(value=str(self.main_interface.stereo.getDisp12MaxDiff()))),
            ("uniquenessRatio", tk.StringVar(value=str(self.main_interface.stereo.getUniquenessRatio()))),
            ("speckleWindowSize", tk.StringVar(value=str(self.main_interface.stereo.getSpeckleWindowSize()))),
            ("speckleRange", tk.StringVar(value=str(self.main_interface.stereo.getSpeckleRange())))
        ]

        self.param_vars = {}
        for param_name, var in params:
            frame = ttk.Frame(root)
            frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(frame, text=param_name).pack(side="left")
            ttk.Entry(frame, textvariable=var, width=10).pack(side="right")
            self.param_vars[param_name] = var

        ttk.Button(root, text="Зберегти", command=self.save_settings).pack(pady=10)

    def save_settings(self):
        try:
            self.main_interface.min_disp = int(self.param_vars["min_disp"].get())
            self.main_interface.num_disp = int(self.param_vars["num_disp"].get())
            self.main_interface.block_size = int(self.param_vars["block_size"].get())
            
            P1_factor = int(self.param_vars["P1_factor"].get())
            P2_factor = int(self.param_vars["P2_factor"].get())
            P1 = P1_factor * 3 * self.main_interface.block_size ** 2
            P2 = P2_factor * 3 * self.main_interface.block_size ** 2

            self.main_interface.stereo = cv2.StereoSGBM_create(
                minDisparity=self.main_interface.min_disp,
                numDisparities=self.main_interface.num_disp,
                blockSize=self.main_interface.block_size,
                P1=P1,
                P2=P2,
                disp12MaxDiff=int(self.param_vars["disp12MaxDiff"].get()),
                uniquenessRatio=int(self.param_vars["uniquenessRatio"].get()),
                speckleWindowSize=int(self.param_vars["speckleWindowSize"].get()),
                speckleRange=int(self.param_vars["speckleRange"].get())
            )

            if self.main_interface.db_connection:
                with self.main_interface.db_connection.cursor() as cursor:
                    params_to_save = {
                        "min_disp": self.main_interface.min_disp,
                        "num_disp": self.main_interface.num_disp,
                        "block_size": self.main_interface.block_size,
                        "P1": P1,
                        "P2": P2,
                        "disp12MaxDiff": int(self.param_vars["disp12MaxDiff"].get()),
                        "uniquenessRatio": int(self.param_vars["uniquenessRatio"].get()),
                        "speckleWindowSize": int(self.param_vars["speckleWindowSize"].get()),
                        "speckleRange": int(self.param_vars["speckleRange"].get())
                    }
                    for param_name, param_value in params_to_save.items():
                        cursor.execute(
                            sql.SQL("""
                                INSERT INTO stereo_params (param_name, param_value, updated_at)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (param_name) 
                                DO UPDATE SET param_value = EXCLUDED.param_value, updated_at = EXCLUDED.updated_at
                            """),
                            [param_name, param_value, datetime.now()]
                        )
                    self.main_interface.db_connection.commit()
                    print("Параметри збережено в БД")

            print("Параметри оновлено:")
            for param_name, var in self.param_vars.items():
                print(f"{param_name}: {var.get()}")
            self.root.destroy()

        except ValueError as e:
            print(f"Помилка: введіть коректні числові значення ({e})")
        except psycopg2.Error as e:
            print(f"Помилка при збереженні до БД: {e}")
            if self.main_interface.db_connection:
                self.main_interface.db_connection.rollback()

class MainInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Стерео обробка")
        self.root.geometry("500x450")

        self.db_connection = get_connection()
        
        self.left_dir = 'Left'
        self.right_dir = 'Right'
        self.output_dir = 'Output'
        os.makedirs(self.output_dir, exist_ok=True)

        self.load_initial_params()

        try:
            icon = tk.PhotoImage(file=r"C:\Users\nazya\Desktop\StereoSceneAI\icon.png")
            self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Помилка завантаження іконки: {e}")
            
        self.root.update_idletasks()
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        x = (width // 2) - (500 // 2)
        y = (height // 2) - (450 // 2)
        self.root.geometry(f"500x450+{x}+{y}")

        self.left_images = []
        self.right_images = []
        self.left_video = None
        self.right_video = None
        self.settings_window = None
        self.is_viewing = False
        self.is_creating = False

        self.check_left = tk.BooleanVar(value=False)
        self.check_right = tk.BooleanVar(value=False)
        self.check_depth = tk.BooleanVar(value=False)

        self.check_left.trace('w', lambda *args: print(f"Стан Ліва камера: {self.check_left.get()}"))
        self.check_right.trace('w', lambda *args: print(f"Стан Права камера: {self.check_right.get()}"))
        self.check_depth.trace('w', lambda *args: print(f"Стан Карта глибин: {self.check_depth.get()}"))

        self.combo_var = tk.StringVar(value="Чорно-білий")

        self.notebook = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Зображення")
        self.notebook.add(self.tab2, text="Відео")
        self.notebook.add(self.tab3, text="Реальний час")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.status_var = tk.StringVar(value="Готово")
        self.status_label = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.create_content(self.tab1)
        self.create_content(self.tab2)
        self.create_realtime_tab(self.tab3)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_initial_params(self):
        default_params = {
            "min_disp": 0,
            "num_disp": 64,
            "block_size": 2,
            "P1": 8 * 2 ** 2,
            "P2": 32 * 2 ** 2,
            "disp12MaxDiff": 5,
            "uniquenessRatio": 5,
            "speckleWindowSize": 5,
            "speckleRange": 32
        }

        if self.db_connection:
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("SELECT param_name, param_value FROM stereo_params")
                    rows = cursor.fetchall()
                    for param_name, param_value in rows:
                        default_params[param_name] = param_value
            except psycopg2.Error as e:
                print(f"Помилка при зчитуванні параметрів із БД: {e}")

        self.min_disp = default_params["min_disp"]
        self.num_disp = default_params["num_disp"]
        self.block_size = max(default_params["block_size"], 1)

        try:
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=self.min_disp,
                numDisparities=self.num_disp,
                blockSize=self.block_size,
                P1=default_params["P1"],
                P2=default_params["P2"],
                disp12MaxDiff=default_params["disp12MaxDiff"],
                uniquenessRatio=default_params["uniquenessRatio"],
                speckleWindowSize=default_params["speckleWindowSize"],
                speckleRange=default_params["speckleRange"]
            )
        except cv2.error as e:
            print(f"Помилка ініціалізації StereoSGBM: {e}")
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=64,
                blockSize=2
            )

    def save_images_to_db(self, left_paths, right_paths):
        if self.db_connection:
            try:
                with self.db_connection.cursor() as cursor:
                    for left_path, right_path in zip(left_paths, right_paths):
                        cursor.execute(
                            sql.SQL("INSERT INTO images (left_filename, right_filename, upload_time) VALUES (%s, %s, %s)"),
                            [left_path, right_path, datetime.now()]
                        )
                    self.db_connection.commit()
                    print(f"Збережено {len(left_paths)} пар зображень до БД")
            except psycopg2.Error as e:
                print(f"Помилка при збереженні до БД: {e}")
                self.db_connection.rollback()

    def save_videos_to_db(self, left_path, right_path):
        if self.db_connection:
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT id FROM videos WHERE left_filename = %s AND right_filename = %s",
                        (left_path, right_path)
                    )
                    result = cursor.fetchone()
                    if not result:
                        cursor.execute(
                            sql.SQL("INSERT INTO videos (left_filename, right_filename, upload_time) VALUES (%s, %s, %s) RETURNING id"),
                            [left_path, right_path, datetime.now()]
                        )
                        video_id = cursor.fetchone()[0]
                        self.db_connection.commit()
                        print(f"Збережено пару відео до БД: {left_path}, {right_path}")
                        return video_id
                    else:
                        return result[0]
            except psycopg2.Error as e:
                print(f"Помилка при збереженні відео до БД: {e}")
                self.db_connection.rollback()
                return None

    def create_content(self, tab):
        input_frame = ttk.LabelFrame(tab, text="Введення даних", padding=5)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        options_frame = ttk.LabelFrame(tab, text="Опції", padding=5)
        options_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        actions_frame = ttk.LabelFrame(tab, text="Дії", padding=5)
        actions_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        if tab == self.tab1:
            ttk.Button(input_frame, text="Ліве зображення", command=self.add_left_images).grid(row=0, column=0, padx=5, pady=5)
            ttk.Button(input_frame, text="Праве зображення", command=self.add_right_images).grid(row=0, column=1, padx=5, pady=5)
        elif tab == self.tab2:
            ttk.Button(input_frame, text="Ліва камера", command=self.add_left_video).grid(row=0, column=0, padx=5, pady=5)
            ttk.Button(input_frame, text="Права камера", command=self.add_right_video).grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(options_frame, text="Параметри", command=self.toggle_settings).grid(row=0, column=0, padx=5, pady=5)
        combo = ttk.Combobox(options_frame, textvariable=self.combo_var, values=["Чорно-білий", "Кольоровий"], state="readonly")
        combo.grid(row=0, column=1, padx=5, pady=5)

        if tab == self.tab2:
            ttk.Checkbutton(options_frame, text="Ліва камера", variable=self.check_left).grid(row=1, column=0, padx=5, pady=2, sticky="w")
            ttk.Checkbutton(options_frame, text="Права камера", variable=self.check_right).grid(row=2, column=0, padx=5, pady=2, sticky="w")
        else:
            ttk.Checkbutton(options_frame, text="Ліве зображення", variable=self.check_left).grid(row=1, column=0, padx=5, pady=2, sticky="w")
            ttk.Checkbutton(options_frame, text="Праве зображення", variable=self.check_right).grid(row=2, column=0, padx=5, pady=2, sticky="w")
        
        ttk.Checkbutton(options_frame, text="Карта глибин", variable=self.check_depth).grid(row=3, column=0, padx=5, pady=2, sticky="w")

        ttk.Button(actions_frame, text="Створити карту глибин", command=self.create_depth_map).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(actions_frame, text="Перегляд карт глибин", command=self.view_depth_maps).grid(row=0, column=1, padx=5, pady=5)
        
        if tab == self.tab1:
            ttk.Button(actions_frame, text="Створити відео", command=self.create_video).grid(row=1, column=0, padx=5, pady=5)

        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

    def create_realtime_tab(self, tab):
        input_frame = ttk.LabelFrame(tab, text="Введення даних", padding=5)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        options_frame = ttk.LabelFrame(tab, text="Опції", padding=5)
        options_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        actions_frame = ttk.LabelFrame(tab, text="Дії", padding=5)
        actions_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        # Кнопка пошуку камер
        ttk.Button(input_frame, text="Пошук камер", command=self.find_cameras).grid(row=0, column=0, padx=5, pady=5)

        # Вибір камер
        ttk.Label(input_frame, text="Ліва камера:").grid(row=1, column=0, padx=5, pady=5)
        self.left_camera_var = tk.StringVar()
        self.left_camera_dropdown = ttk.Combobox(input_frame, textvariable=self.left_camera_var)
        self.left_camera_dropdown.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Права камера:").grid(row=2, column=0, padx=5, pady=5)
        self.right_camera_var = tk.StringVar()
        self.right_camera_dropdown = ttk.Combobox(input_frame, textvariable=self.right_camera_var)
        self.right_camera_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Опції
        ttk.Button(options_frame, text="Параметри", command=self.toggle_settings).grid(row=0, column=0, padx=5, pady=5)
        combo = ttk.Combobox(options_frame, textvariable=self.combo_var, values=["Чорно-білий", "Кольоровий"], state="readonly")
        combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Checkbutton(options_frame, text="Ліва камера", variable=self.check_left).grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(options_frame, text="Права камера", variable=self.check_right).grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(options_frame, text="Карта глибин", variable=self.check_depth).grid(row=3, column=0, padx=5, pady=2, sticky="w")

        # Дії
        ttk.Button(actions_frame, text="Запустити реальний час", command=self.start_realtime).grid(row=0, column=0, padx=5, pady=5)

        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

    def find_cameras(self):
        """Пошук доступних камер і заповнення dropdown-меню."""
        available_cameras = []
        for i in range(10):  # Перевіряємо перші 10 індексів
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        if available_cameras:
            self.left_camera_dropdown['values'] = available_cameras
            self.right_camera_dropdown['values'] = available_cameras
            self.status_var.set(f"Знайдено камери: {available_cameras}")
        else:
            self.status_var.set("Камери не знайдено.")
            messagebox.showwarning("Попередження", "Камери не знайдено.")

    def start_realtime(self):
        """Запуск обробки в реальному часі."""
        if self.is_creating:
            self.status_var.set("Процес уже запущено")
            return

        self.is_creating = True
        self.status_var.set("Запуск реального часу...")

        left_index = int(self.left_camera_var.get()) if self.left_camera_var.get() else 0
        right_index = int(self.right_camera_var.get()) if self.right_camera_var.get() else 1

        left_cap = cv2.VideoCapture(left_index)
        right_cap = cv2.VideoCapture(right_index)

        if not left_cap.isOpened() or not right_cap.isOpened():
            self.status_var.set("Не вдалося відкрити камери.")
            self.is_creating = False
            return

        # Ініціалізація відеозапису
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mode = self.combo_var.get()
        output_filename = f"realtime_video_{'black' if mode == 'Чорно-білий' else 'color'}_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        frame_size = (640, 480)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Збереження відео в таблицю videos
        video_id = self.save_videos_to_db("realtime_left", "realtime_right")
        if video_id is None:
            self.status_var.set("Помилка: не вдалося зберегти відео в БД")
            self.is_creating = False
            left_cap.release()
            right_cap.release()
            video_writer.release()
            return

        start_time = datetime.now()

        # Отримуємо розміри екрану
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        while self.is_creating:
            ret_left, left_frame = left_cap.read()
            ret_right, right_frame = right_cap.read()

            if not ret_left or not ret_right:
                self.status_var.set("Не вдалося отримати кадри з камер.")
                break

            # Зміна розміру для покращення продуктивності
            left_frame = cv2.resize(left_frame, frame_size)
            right_frame = cv2.resize(right_frame, frame_size)

            # Перетворення в сірий формат
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # Обчислення карти диспаратності
            disparity_map = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            mode = self.combo_var.get()
            if mode == "Чорно-білий":
                disparity_map_gray = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                output_map = cv2.cvtColor(disparity_map_gray, cv2.COLOR_GRAY2BGR)
            else:
                disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
                output_map = cv2.applyColorMap(np.uint8(disparity_map_normalized), cv2.COLORMAP_JET)

            # Відображення результатів
            # Лівий верхній кут для "Left Camera"
            left_x = 10  # Невеликий відступ від лівого краю
            left_y = 10  # Невеликий відступ від верхнього краю

            # Правий верхній кут для "Right Camera"
            right_x = screen_width - frame_size[0] - 10  # Відступ від правого краю
            right_y = 10  # Невеликий відступ від верхнього краю

            # Центр для "Depth Map"
            center_x = screen_width // 2 - frame_size[0] // 2
            center_y = screen_height // 2 - frame_size[1] // 2 + 100

            if self.check_left.get():
                cv2.imshow("Left Camera", left_frame)
                cv2.moveWindow("Left Camera", left_x, left_y)
            if self.check_right.get():
                cv2.imshow("Right Camera", right_frame)
                cv2.moveWindow("Right Camera", right_x, right_y)
            if self.check_depth.get():
                cv2.imshow("Depth Map", output_map)
                cv2.moveWindow("Depth Map", center_x, center_y)

            # Запис у відео
            video_writer.write(output_map)

            # Натискання 'q' для виходу
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_creating = False
                break

        process_time = (datetime.now() - start_time).total_seconds()
        left_cap.release()
        right_cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        if self.is_creating:
            self.save_video_depth_map_to_db(video_id, video_id, output_path, process_time)
            self.status_var.set(f"Реальний час завершено. Відео збережено: {output_path}")
        else:
            self.status_var.set("Реальний час перервано")
            if os.path.exists(output_path):
                os.remove(output_path)

        self.is_creating = False

    def add_left_images(self):
        folder = filedialog.askdirectory(title="Виберіть папку з лівими зображеннями")
        if folder:
            self.left_images = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if self.left_images:
                self.status_var.set(f"Вибрано {len(self.left_images)} лівих зображень")
                print(f"Знайдено {len(self.left_images)} лівих зображень")
            else:
                self.status_var.set("У папці немає зображень")

    def add_right_images(self):
        folder = filedialog.askdirectory(title="Виберіть папку з правими зображеннями")
        if folder:
            self.right_images = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if self.right_images:
                self.status_var.set(f"Вибрано {len(self.right_images)} правих зображень")
                print(f"Знайдено {len(self.right_images)} правих зображень")
                if self.left_images and len(self.left_images) == len(self.right_images):
                    self.save_images_to_db(self.left_images, self.right_images)
                else:
                    self.status_var.set("Помилка: виберіть однакову кількість зображень для обох сторін")
            else:
                self.status_var.set("У папці немає зображень")

    def add_left_video(self):
        file = filedialog.askopenfilename(title="Виберіть ліве відео", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file:
            self.left_video = file
            self.status_var.set(f"Ліва камера: {os.path.basename(file)}")
            print("Вибране відео для лівої камери:", file)

    def add_right_video(self):
        file = filedialog.askopenfilename(title="Виберіть праве відео", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file:
            self.right_video = file
            self.status_var.set(f"Права камера: {os.path.basename(file)}")
            print("Вибране відео для правої камери:", file)

    def toggle_settings(self):
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.destroy()
            self.status_var.set("Вікно параметрів закрито")
        else:
            self.settings_window = tk.Toplevel(self.root)
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
            self.settings_window.geometry(f"300x400+{main_x - 320}+{main_y}")
            SettingsInterface(self.settings_window, self)
            self.status_var.set("Відкрито вікно параметрів")

    def create_depth_map(self):
        if not self.db_connection:
            self.status_var.set("Помилка: немає підключення до БД")
            return

        if self.is_creating:
            self.status_var.set("Процес створення вже запущено")
            return

        self.is_creating = True
        self.status_var.set("Створення карт глибин...")
        cv2.destroyAllWindows()

        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab == "Зображення":
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, left_filename, right_filename 
                        FROM images 
                        ORDER BY upload_time ASC
                    """)
                    image_pairs = cursor.fetchall()

                    if not image_pairs:
                        self.status_var.set("Немає зображень у базі даних")
                        self.is_creating = False
                        return

                    print(f"Знайдено {len(image_pairs)} пар зображень у БД")
                    start_time = datetime.now()
                    for idx, (image_id, left_img_path, right_img_path) in enumerate(image_pairs):
                        if not self.is_creating:
                            break

                        left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
                        right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)

                        if left_img is None or right_img is None:
                            continue

                        # Перетворення в сірий формат
                        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

                        # Обчислення карти диспаратності
                        disparity_map = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

                        # Фільтруємо недійсні значення (замінюємо -1 на мінімальне дійсне значення)
                        disparity_map[disparity_map == -1] = np.min(disparity_map[disparity_map != -1])

                        # Нормалізація карти диспаратності
                        mode = self.combo_var.get()
                        disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        # Створюємо карту для відображення та збереження
                        if mode == "Чорно-білий":
                            output_map = cv2.cvtColor(disparity_map_normalized, cv2.COLOR_GRAY2BGR)
                        else:
                            output_map = cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_JET)

                        # Збереження карти глибини
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        base_filename = os.path.basename(left_img_path).split('.')[0]
                        filename = f"disparity_{base_filename}_{'black' if mode == 'Чорно-білий' else 'color'}_{timestamp}.png"
                        output_path = os.path.join(self.output_dir, filename)
                        cv2.imwrite(output_path, output_map)

                        process_time = (datetime.now() - start_time).total_seconds()
                        self.save_image_depth_map_to_db(image_id, image_id, output_path, process_time)

                        # Відображення
                        cv2.destroyAllWindows()
                        # Отримуємо розміри екрану
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()

                        # Лівий верхній кут для "Left Image"
                        left_x = 10  # Невеликий відступ від лівого краю
                        left_y = 10  # Невеликий відступ від верхнього краю

                        # Правий верхній кут для "Right Image"
                        right_x = screen_width - right_img.shape[1] - 10  # Відступ від правого краю
                        right_y = 10  # Невеликий відступ від верхнього краю

                        # Центр для "Depth Map"
                        center_x = screen_width // 2 - output_map.shape[1] // 2
                        center_y = screen_height // 2 - output_map.shape[0] // 2 + 100

                        if self.check_depth.get():
                            cv2.imshow("Depth Map", output_map)
                            cv2.moveWindow("Depth Map", center_x, center_y)
                        if self.check_left.get():
                            cv2.imshow("Left Image", left_img)
                            cv2.moveWindow("Left Image", left_x, left_y)
                        if self.check_right.get():
                            cv2.imshow("Right Image", right_img)
                            cv2.moveWindow("Right Image", right_x, right_y)

                        if cv2.waitKey(1000) & 0xFF == ord('q'):
                            self.is_creating = False
                            break

                        self.root.after(0, lambda: self.status_var.set(f"Оброблено {idx + 1}/{len(image_pairs)} зображень"))

                    if self.is_creating:
                        self.root.after(0, lambda: self.status_var.set("Карти глибин створено та збережено в БД"))
                    else:
                        self.root.after(0, lambda: self.status_var.set("Обробку перервано"))

            except psycopg2.Error as e:
                self.root.after(0, lambda msg=str(e): self.status_var.set(f"Помилка БД: {msg}"))
            finally:
                self.is_creating = False
                cv2.destroyAllWindows()

        elif current_tab == "Відео":
            if not self.left_video or not self.right_video:
                self.status_var.set("Помилка: виберіть обидва відеофайли")
                self.is_creating = False
                return

            left_cap = cv2.VideoCapture(self.left_video)
            right_cap = cv2.VideoCapture(self.right_video)

            if not left_cap.isOpened() or not right_cap.isOpened():
                self.status_var.set("Помилка: не вдалося відкрити відеофайли")
                self.is_creating = False
                return

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            mode = self.combo_var.get()
            output_filename = f"disparity_video_{'black' if mode == 'Чорно-білий' else 'color'}_{timestamp}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            frame_size = (640, 480)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

            video_id = self.save_videos_to_db(self.left_video, self.right_video)
            if video_id is None:
                self.status_var.set("Помилка: не вдалося зберегти відео в БД")
                self.is_creating = False
                left_cap.release()
                right_cap.release()
                video_writer.release()
                return

            start_time = datetime.now()
            frame_count = 0

            # Отримуємо розміри екрану
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            while left_cap.isOpened() and right_cap.isOpened() and self.is_creating:
                ret_left, left_frame = left_cap.read()
                ret_right, right_frame = right_cap.read()

                if not ret_left or not ret_right:
                    print("[ІНФО] Кінець відео.")
                    break

                left_frame = cv2.resize(left_frame, frame_size)
                right_frame = cv2.resize(right_frame, frame_size)

                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                disparity_map = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

                mode = self.combo_var.get()
                if mode == "Чорно-білий":
                    disparity_map_gray = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    output_map = cv2.cvtColor(disparity_map_gray, cv2.COLOR_GRAY2BGR)
                else:
                    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
                    output_map = cv2.applyColorMap(np.uint8(disparity_map_normalized), cv2.COLORMAP_JET)

                video_writer.write(output_map)
                frame_count += 1

                # Лівий верхній кут для "Left Camera"
                left_x = 10  # Невеликий відступ від лівого краю
                left_y = 10  # Невеликий відступ від верхнього краю

                # Правий верхній кут для "Right Camera"
                right_x = screen_width - frame_size[0] - 10  # Відступ від правого краю
                right_y = 10  # Невеликий відступ від верхнього краю

                # Центр для "Depth Map"
                center_x = screen_width // 2 - frame_size[0] // 2
                center_y = screen_height // 2 - frame_size[1] // 2 + 100

                if self.check_depth.get():
                    cv2.imshow("Depth Map", output_map)
                    cv2.moveWindow("Depth Map", center_x, center_y)
                if self.check_left.get():
                    cv2.imshow("Left Camera", left_frame)
                    cv2.moveWindow("Left Camera", left_x, left_y)
                if self.check_right.get():
                    cv2.imshow("Right Camera", right_frame)
                    cv2.moveWindow("Right Camera", right_x, right_y)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_creating = False
                    break

                self.root.after(0, lambda: self.status_var.set(f"Оброблено {frame_count} кадрів"))

            process_time = (datetime.now() - start_time).total_seconds()
            video_writer.release()
            left_cap.release()
            right_cap.release()

            if self.is_creating and frame_count > 0:
                self.save_video_depth_map_to_db(video_id, video_id, output_path, process_time)
                self.root.after(0, lambda: self.status_var.set(f"Карта глибини для відео створена та збережена: {output_path}"))
            else:
                self.root.after(0, lambda: self.status_var.set("Обробку перервано або помилка"))
                if os.path.exists(output_path) and frame_count == 0:
                    os.remove(output_path)

            self.is_creating = False
            cv2.destroyAllWindows()

        elif current_tab == "Реальний час":
            self.start_realtime()

    def save_image_depth_map_to_db(self, left_image_id, right_image_id, output_path, process_time):
        if self.db_connection:
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO image_depth_maps 
                        (left_image_id, right_image_id, filename, process_time, report, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (left_image_id, right_image_id, output_path, process_time, "Success", datetime.now())
                    )
                    self.db_connection.commit()
                    print(f"Image depth map saved to DB: {output_path}")
            except psycopg2.Error as e:
                print(f"Помилка при збереженні до БД: {e}")
                self.db_connection.rollback()

    def save_video_depth_map_to_db(self, left_video_id, right_video_id, output_path, process_time):
        if self.db_connection:
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO video_depth_maps 
                        (left_video_id, right_video_id, filename, process_time, report, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (left_video_id, right_video_id, output_path, process_time, "Success", datetime.now())
                    )
                    self.db_connection.commit()
                    print(f"Video depth map saved to DB: {output_path}")
            except psycopg2.Error as e:
                print(f"Помилка при збереженні до БД: {e}")
                self.db_connection.rollback()

    def view_depth_maps(self):
        if not self.db_connection:
            self.status_var.set("Помилка: немає підключення до БД")
            return

        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab == "Зображення":
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT idm.filename, i.left_filename, i.right_filename
                        FROM image_depth_maps idm
                        JOIN images i ON idm.left_image_id = i.id
                        ORDER BY idm.created_at DESC
                    """)
                    depth_maps = cursor.fetchall()
                    
                    if not depth_maps:
                        self.status_var.set("Немає збережених карт глибин")
                        return
                    
                    self.is_viewing = True
                    current_index = 0
                    
                    def show_image(index):
                        if not self.is_viewing or index < 0 or index >= len(depth_maps):
                            return
                        filename, left_filename, right_filename = depth_maps[index]
                        depth_map = cv2.imread(filename)
                        left_img = cv2.imread(left_filename, cv2.IMREAD_COLOR)
                        right_img = cv2.imread(right_filename, cv2.IMREAD_COLOR)

                        cv2.destroyAllWindows()
                        # Отримуємо розміри екрану
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()

                        # Лівий верхній кут для "Left Image"
                        left_x = 10  # Невеликий відступ від лівого краю
                        left_y = 10  # Невеликий відступ від верхнього краю

                        # Правий верхній кут для "Right Image"
                        right_x = screen_width - right_img.shape[1] - 10  # Відступ від правого краю
                        right_y = 10  # Невеликий відступ від верхнього краю

                        # Центр для "Depth Map"
                        center_x = screen_width // 2 - depth_map.shape[1] // 2
                        center_y = screen_height // 2 - depth_map.shape[0] // 2

                        if self.check_depth.get() and depth_map is not None:
                            cv2.imshow("Depth Map", depth_map)
                            cv2.moveWindow("Depth Map", center_x, center_y)
                        if self.check_left.get() and left_img is not None:
                            cv2.imshow("Left Image", left_img)
                            cv2.moveWindow("Left Image", left_x, left_y)
                        if self.check_right.get() and right_img is not None:
                            cv2.imshow("Right Image", right_img)
                            cv2.moveWindow("Right Image", right_x, right_y)

                    show_image(current_index)
                    
                    while self.is_viewing:
                        key = cv2.waitKeyEx(100)
                        if key == 2424832:  # Стрілка вліво
                            current_index = max(current_index - 1, 0)
                            show_image(current_index)
                        elif key == 2555904:  # Стрілка вправо
                            current_index = min(current_index + 1, len(depth_maps) - 1)
                            show_image(current_index)
                        elif key == 27:  # Esc
                            self.is_viewing = False
                            cv2.destroyAllWindows()
                            break
                    
                    self.status_var.set("Перегляд завершено")
            except psycopg2.Error as e:
                self.status_var.set(f"Помилка при отриманні карт глибин: {e}")
            finally:
                self.is_viewing = False

        elif current_tab in ["Відео", "Реальний час"]:
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT vdm.filename
                        FROM video_depth_maps vdm
                        JOIN videos v ON vdm.left_video_id = v.id
                        ORDER BY vdm.created_at DESC
                        LIMIT 1
                    """)
                    result = cursor.fetchone()
                    
                    if not result:
                        self.status_var.set("Немає збережених карт глибин для відео")
                        return
                    
                    filename = result[0]
                    cap = cv2.VideoCapture(filename)
                    
                    if not cap.isOpened():
                        self.status_var.set("Помилка: не вдалося відкрити відео карти глибини")
                        return

                    self.is_viewing = True

                    while cap.isOpened() and self.is_viewing:
                        ret, depth_frame = cap.read()
                        if not ret:
                            break

                        # Отримуємо розміри екрану
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()

                        # Центр для "Depth Map"
                        center_x = screen_width // 2 - depth_frame.shape[1] // 2
                        center_y = screen_height // 2 - depth_frame.shape[0] // 2

                        cv2.imshow("Depth Map", depth_frame)
                        cv2.moveWindow("Depth Map", center_x, center_y)

                        key = cv2.waitKey(33) & 0xFF
                        if key == ord('q') or key == 27:
                            self.is_viewing = False
                            break

                    cap.release()
                    cv2.destroyAllWindows()
                    self.status_var.set("Перегляд завершено")

            except psycopg2.Error as e:
                self.status_var.set(f"Помилка при отриманні карт глибин: {e}")
            finally:
                self.is_viewing = False

    def create_video(self):
        self.status_var.set("Створення відео...")
        disparity_files = sorted([f for f in os.listdir(self.output_dir) if f.startswith('disparity_')])

        if not disparity_files:
            self.status_var.set("Не знайдено карт глибин для відео.")
            return

        first_frame_path = os.path.join(self.output_dir, disparity_files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            self.status_var.set(f"Помилка: не вдалося завантажити {first_frame_path}")
            return

        height, width = first_frame.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_output_path = os.path.join(self.output_dir, f'disparity_video_{timestamp}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

        for disparity_file in disparity_files:
            frame_path = os.path.join(self.output_dir, disparity_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            video_writer.write(frame)

        video_writer.release()
        self.status_var.set(f"Відео збережено як: {video_output_path}")

    def on_closing(self):
        self.is_viewing = False
        self.is_creating = False
        if self.db_connection:
            close_connection(self.db_connection)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainInterface(root)
    root.mainloop()