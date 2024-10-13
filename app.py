import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import model
import camera

class App:
    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window_title = window_title
        self.counters = [1, 1]  # Counters to track saved images
        self.model = model.Model()
        self.auto_predict = False
        self.camera = camera.Camera()

        self.init_gui()
        self.delay = 15
        self.update()  # Start the update loop
        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=self.train_and_enable_prediction)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict, state=tk.DISABLED)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS", font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if ret:
            os.makedirs(str(class_num), exist_ok=True)  # Ensure folder exists
            file_path = f'{class_num}/frame{self.counters[class_num - 1]}.jpg'
            cv.imwrite(file_path, cv.cvtColor(frame, cv.COLOR_RGB2GRAY))

            img = PIL.Image.open(file_path)
            img.thumbnail((150, 150), PIL.Image.LANCZOS)  # Resize for saving
            img.save(file_path)

            self.counters[class_num - 1] += 1

    def train_and_enable_prediction(self):
        success = self.model.train_model(self.counters)
        if success:
            self.btn_predict.config(state=tk.NORMAL)
            messagebox.showinfo("Info", "Model trained successfully!")

    def predict(self):
        ret, frame = self.camera.get_frame()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame.")
            return

        try:
            prediction = self.model.predict(frame)
            if prediction == 1:
                self.class_label.config(text=self.classname_one)
            elif prediction == 2:
                self.class_label.config(text=self.classname_two)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def reset(self):
        for folder in ['1', '2']:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    os.remove(os.path.join(folder, file))
        self.counters = [1, 1]
        self.model = model.Model()
        self.class_label.config(text="CLASS")
        self.btn_predict.config(state=tk.DISABLED)

    def update(self):
        if self.auto_predict:
            self.predict()

        ret, frame = self.camera.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)
