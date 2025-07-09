import tensorflow as tf
import numpy as np
from tkinter import filedialog, Tk, Label, Button, Frame
from PIL import Image, ImageTk
import os

model = tf.keras.models.load_model("plant_disease_model.h5")
class_names = ['Healthy 🍃', 'Rust 🍂', 'Powdery 🌫️']

def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        try:
            img = Image.open(file_path).convert("RGB").resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = np.max(prediction)

            display_img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(display_img)
            panel.config(image=img_tk)
            panel.image = img_tk

            result_label.config(text=f"🧪 Prediction: {predicted_class}\n🔍 Confidence: {confidence:.2%}")

        except Exception as e:
            result_label.config(text=f"❌ Error: {e}")

window = Tk()
window.title("🌱 Smart Plant Doctor")
window.geometry("420x500")
window.configure(bg="#f0f8f0")

title_label = Label(window, text="🌿 Plant Disease Classifier 🌿", font=("Helvetica", 16, "bold"), bg="#f0f8f0", fg="#2e8b57")
title_label.pack(pady=10)

panel = Label(window, bg="#f0f8f0")
panel.pack(pady=10)

upload_btn = Button(window, text="📁 Upload Plant Leaf", command=upload_and_predict, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
upload_btn.pack(pady=10)

result_label = Label(window, text="", font=("Arial", 13), bg="#f0f8f0", fg="#333")
result_label.pack(pady=10)

footer_label = Label(window, text="🔬 Powered by AI | Developed with ❤️", font=("Arial", 9), bg="#f0f8f0", fg="#888")
footer_label.pack(side="bottom", pady=10)

window.mainloop()
