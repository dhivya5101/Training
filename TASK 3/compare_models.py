import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.lite.python.interpreter import Interpreter

base_dir = "dataset"
test_dir = os.path.join(base_dir, "test")

img_size = (150, 150)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

model = load_model('plant_disease_model.h5')

tflite_model_path = 'converted_model.tflite'
if not os.path.exists(tflite_model_path):
    print("Converting Keras model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model successfully converted to {tflite_model_path}")

def evaluate_keras(model, generator):
    start = time.time()
    predictions = model.predict(generator)
    end = time.time()

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = generator.classes
    accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
    return accuracy, end - start

def evaluate_tflite(model_path, generator):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    correct = 0
    total = 0
    start = time.time()
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        for j in range(x_batch.shape[0]):
            input_data = np.expand_dims(x_batch[j], axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            if np.argmax(output) == np.argmax(y_batch[j]):
                correct += 1
            total += 1
    end = time.time()

    accuracy = correct / total
    return accuracy, end - start

def get_model_size(model_path):
    return os.path.getsize(model_path)

keras_acc, keras_time = evaluate_keras(model, test_generator)
tflite_acc, tflite_time = evaluate_tflite(tflite_model_path, test_generator)

keras_size = get_model_size('plant_disease_model.h5')
tflite_size = get_model_size(tflite_model_path)

data = {
    "Model": ["Keras", "TensorFlow Lite"],
    "Size (bytes)": [keras_size, tflite_size],
    "Accuracy (%)": [keras_acc * 100, tflite_acc * 100],
    "Inference Time (s)": [keras_time, tflite_time]
}

df = pd.DataFrame(data)

print("\nModel Comparison:")
print(df)

print("\n🔹 **Trade-Off Notes:**")
print("1. **Keras model** typically has higher accuracy but is larger in size and slower during inference.")
print("2. **TensorFlow Lite model** is much smaller in size, making it ideal for deployment on mobile/edge devices, but may have slightly lower accuracy and slower inference time due to optimizations for speed and size reduction.")