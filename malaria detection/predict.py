import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("malaria_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    prediction = model.predict(img)
    return "Infected" if prediction[0] > 0.5 else "Uninfected"

# Test prediction
image_path = "cell_images/u1.png"
result = predict_image(image_path)
print(f"Prediction: {result}")
