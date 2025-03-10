import tkinter as tk
from tkinter import ttk

class SettingsInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Параметри")
        self.root.geometry("300x400")
        # Додавання іконки
        try:
            icon = tk.PhotoImage(file=r"C:\Users\nazya\Desktop\StereoSceneAI\icon.png")
            self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Помилка завантаження іконки: {e}")
            
        # Заголовок
        ttk.Label(root, text="Параметри", anchor="center").pack(pady=10)

        # Поля для введення параметрів
        params = [
            ("min_disp", tk.StringVar(value="0")),
            ("num_disp", tk.StringVar(value="64")),
            ("block_size", tk.StringVar(value="2")),
            ("P1", tk.StringVar(value="24")),  # 4 * 3 * block_size ** 2
            ("P2", tk.StringVar(value="96")),  # 16 * 3 * block_size ** 2
            ("disp12MaxDiff", tk.StringVar(value="5")),
            ("uniquenessRatio", tk.StringVar(value="5")),
            ("speckleWindowSize", tk.StringVar(value="5")),
            ("speckleRange", tk.StringVar(value="32"))
        ]

        for param_name, var in params:
            frame = ttk.Frame(root)
            frame.pack(fill="x", padx=5, pady=2)
            ttk.Label(frame, text=param_name).pack(side="left")
            ttk.Entry(frame, textvariable=var, width=10).pack(side="right")

        # Кнопка збереження
        ttk.Button(root, text="Зберегти", command=self.save_settings).pack(pady=10)

    def save_settings(self):
        print("Параметри збережено:")
        print(f"min_disp: {self.root.children['.!frame.!label2'].children['.!entry'].get()}")
        print(f"num_disp: {self.root.children['.!frame2'].children['.!entry'].get()}")
        print(f"block_size: {self.root.children['.!frame3'].children['.!entry'].get()}")
        print(f"P1: {self.root.children['.!frame4'].children['.!entry'].get()}")
        print(f"P2: {self.root.children['.!frame5'].children['.!entry'].get()}")
        print(f"disp12MaxDiff: {self.root.children['.!frame6'].children['.!entry'].get()}")
        print(f"uniquenessRatio: {self.root.children['.!frame7'].children['.!entry'].get()}")
        print(f"speckleWindowSize: {self.root.children['.!frame8'].children['.!entry'].get()}")
        print(f"speckleRange: {self.root.children['.!frame9'].children['.!entry'].get()}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SettingsInterface(root)
    root.mainloop()