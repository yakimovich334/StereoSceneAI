import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import settings_interface  # Імпорт вікна параметрів

class MainInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Стерео обробка")
        self.root.geometry("500x450")

        # Додавання іконки
        try:
            icon = tk.PhotoImage(file=r"C:\Users\nazya\Desktop\StereoSceneAI\icon.png")
            self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Помилка завантаження іконки: {e}")
            
        # Центрування головного вікна
        self.root.update_idletasks()
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        x = (width // 2) - (500 // 2)
        y = (height // 2) - (450 // 2)
        self.root.geometry(f"500x450+{x}+{y}")

        # Змінні для збереження шляхів до файлів
        self.left_image = None
        self.right_image = None
        self.left_video = None
        self.right_video = None
        self.settings_window = None  # Змінна для збереження вікна параметрів

        # Створення вкладок
        self.notebook = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Зображення")
        self.notebook.add(self.tab2, text="Відео")
        self.notebook.add(self.tab3, text="Реальний час")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Статусний рядок
        self.status_var = tk.StringVar(value="Готово")
        self.status_label = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Налаштування сітки для розтягування
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Елементи для вкладок
        self.create_content(self.tab1)  # Зображення
        self.create_content(self.tab2)  # Відео
        self.create_content(self.tab3)  # Реальний час

    def create_content(self, tab):
        input_frame = ttk.LabelFrame(tab, text="Введення даних", padding=5)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        options_frame = ttk.LabelFrame(tab, text="Опції", padding=5)
        options_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        actions_frame = ttk.LabelFrame(tab, text="Дії", padding=5)
        actions_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        if tab == self.tab1:
            ttk.Button(input_frame, text="Ліве зображення", command=self.add_left_image).grid(row=0, column=0, padx=5, pady=5)
            ttk.Button(input_frame, text="Праве зображення", command=self.add_right_image).grid(row=0, column=1, padx=5, pady=5)
        elif tab == self.tab2:
            ttk.Button(input_frame, text="Ліве відео", command=self.add_left_video).grid(row=0, column=0, padx=5, pady=5)
            ttk.Button(input_frame, text="Праве відео", command=self.add_right_video).grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(options_frame, text="Параметри", command=self.toggle_settings).grid(row=0, column=0, padx=5, pady=5)
        self.combo = ttk.Combobox(options_frame, values=["Чорно-білий", "Кольоровий"], state="readonly")
        self.combo.set("Чорно-білий")
        self.combo.grid(row=0, column=1, padx=5, pady=5)

        self.check_left = tk.BooleanVar()
        self.check_right = tk.BooleanVar()
        self.check_depth = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Ліве зображення", variable=self.check_left).grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(options_frame, text="Праве зображення", variable=self.check_right).grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(options_frame, text="Карта глибин", variable=self.check_depth).grid(row=3, column=0, padx=5, pady=2, sticky="w")

        ttk.Button(actions_frame, text="Створити карту глибин", command=self.create_depth_map).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(actions_frame, text="Перегляд карт глибин", command=self.view_depth_maps).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(actions_frame, text="Створити відео", command=self.create_video).grid(row=1, column=0, padx=5, pady=5)

        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

    def add_left_image(self):
        file = filedialog.askopenfilename(title="Виберіть ліве зображення", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file:
            self.left_image = file
            self.status_var.set(f"Ліве зображення: {file}")
            print("Вибране ліве зображення:", file)

    def add_right_image(self):
        file = filedialog.askopenfilename(title="Виберіть праве зображення", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file:
            self.right_image = file
            self.status_var.set(f"Праве зображення: {file}")
            print("Вибране праве зображення:", file)

    def add_left_video(self):
        file = filedialog.askopenfilename(title="Виберіть ліве відео", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file:
            self.left_video = file
            self.status_var.set(f"Ліве відео: {file}")
            print("Вибране ліве відео:", file)

    def add_right_video(self):
        file = filedialog.askopenfilename(title="Виберіть праве відео", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file:
            self.right_video = file
            self.status_var.set(f"Праве відео: {file}")
            print("Вибране праве відео:", file)

    def toggle_settings(self):
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.destroy()
            self.status_var.set("Вікно параметрів закрито")
        else:
            self.settings_window = tk.Toplevel(self.root)
            # Розташування лівіше від головного вікна
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
            self.settings_window.geometry(f"300x400+{main_x - 320}+{main_y}")
            settings_interface.SettingsInterface(self.settings_window)
            self.status_var.set("Відкрито вікно параметрів")

    def create_depth_map(self):
        if (self.left_image and self.right_image) or (self.left_video and self.right_video):
            print("Створення карти глибин...")
            self.status_var.set("Створення карти глибин...")
        else:
            self.status_var.set("Помилка: виберіть обидва файли")

    def view_depth_maps(self):
        print("Перегляд створених карт глибин...")
        self.status_var.set("Перегляд карт глибин")

    def create_video(self):
        print("Створення відео...")
        self.status_var.set("Створення відео...")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainInterface(root)
    root.mainloop()