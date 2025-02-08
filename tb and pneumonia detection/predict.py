import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("tb_pneumonia_model.h5")

# Class labels
CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

# Predict function
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"ERROR: Image {img_path} does not exist!")
        return

    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Scores: {prediction[0]}")

# Example usage
img_path = r"D:/neuralcare/tb and pneumonia/tb1.png"
predict_image(img_path)
