import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
from tkinter import filedialog, Tk, Label, Button, Frame, messagebox
from PIL import ImageTk
import requests

model = tf.keras.models.load_model("plant_disease_model.h5")
class_names = ['Healthy 🍃', 'Rust 🍂', 'Powdery 🌫️']

disease_info = {
    'Healthy 🍃': "Your plant appears to be healthy! Keep up the good care. 🌱",
    'Rust 🍂': "Rust is caused by fungi, leading to orange-brown spots. Treat with fungicides and remove affected leaves. 🍂",
    'Powdery 🌫️': "Powdery mildew is a fungal disease causing white powdery spots. Improve air circulation and apply fungicides. 🌫️"
}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream).convert("RGB").resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = np.max(prediction)

        return jsonify({
            "prediction": predicted_class,
            "confidence": str(confidence),
            "info": disease_info[predicted_class]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        try:
            result_label.config(text="⏳ Processing...")
            window.update()

            img = Image.open(file_path).convert("RGB").resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            url = "http://127.0.0.1:5000/predict"
            files = {'file': open(file_path, 'rb')}
            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                predicted_class = result['prediction']
                confidence = float(result['confidence'])
                disease_info_text = result['info']

                display_img = img.resize((200, 200))
                img_tk = ImageTk.PhotoImage(display_img)
                panel.config(image=img_tk, borderwidth=2, relief="solid")
                panel.image = img_tk

                result_text = f"🧪 Prediction: {predicted_class}\n🔍 Confidence: {confidence:.2%}\n📜 Info: {disease_info_text}"
                result_label.config(text=result_text)

                save_btn.config(state="normal")
            else:
                result_label.config(text=f"❌ Error: {response.json().get('error')}")
                save_btn.config(state="disabled")
            
        except Exception as e:
            result_label.config(text=f"❌ Error: {e}")
            save_btn.config(state="disabled")

def clear_image():
    panel.config(image='', borderwidth=0, relief="flat")
    panel.image = None
    result_label.config(text="")
    save_btn.config(state="disabled")

def save_prediction():
    prediction_text = result_label.cget("text")
    if prediction_text and "Prediction" in prediction_text:
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(prediction_text)
            messagebox.showinfo("Success", "Prediction saved successfully!")
    else:
        messagebox.showwarning("Warning", "No prediction to save!")

window = Tk()
window.title("🌱 Smart Plant Doctor")
window.geometry("420x600")
window.configure(bg="#f0f8f0")

title_label = Label(window, text="🌿 Plant Disease Classifier 🌿", font=("Helvetica", 18, "bold"), bg="#f0f8f0", fg="#2e8b57")
title_label.pack(pady=15)

panel = Label(window, bg="#f0f8f0")
panel.pack(pady=10)

button_frame = Frame(window, bg="#f0f8f0")
button_frame.pack(pady=10)

def on_enter(e, btn): btn.config(bg="#45a049")
def on_leave(e, btn): btn.config(bg="#4CAF50")

upload_btn = Button(button_frame, text="📁 Upload Plant Leaf", command=upload_and_predict, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
upload_btn.grid(row=0, column=0, padx=5)
upload_btn.bind("<Enter>", lambda e: on_enter(e, upload_btn))
upload_btn.bind("<Leave>", lambda e: on_leave(e, upload_btn))

clear_btn = Button(button_frame, text="🗑️ Clear", command=clear_image, font=("Arial", 12), bg="#f44336", fg="white", padx=10, pady=5)
clear_btn.grid(row=0, column=1, padx=5)
clear_btn.bind("<Enter>", lambda e: on_enter(e, clear_btn))
clear_btn.bind("<Leave>", lambda e: on_leave(e, clear_btn))

save_btn = Button(button_frame, text="💾 Save Prediction", command=save_prediction, font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state="disabled")
save_btn.grid(row=0, column=2, padx=5)
save_btn.bind("<Enter>", lambda e: on_enter(e, save_btn))
save_btn.bind("<Leave>", lambda e: on_leave(e, save_btn))

result_label = Label(window, text="", font=("Arial", 12), bg="#f0f8f0", fg="#333", wraplength=380, justify="left")
result_label.pack(pady=15)

footer_label = Label(window, text="🔬 Powered by AI | Developed with ❤️", font=("Arial", 10, "italic"), bg="#f0f8f0", fg="#888")
footer_label.pack(side="bottom", pady=10)

window.mainloop()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
