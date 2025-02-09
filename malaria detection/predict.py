import cv2
import numpy as np
import tensorflow as tf  

# Load trained model
model = tf.keras.models.load_model(r"D:\neuralcare\malaria detection\malaria_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    
    # Ensure the image was loaded correctly
    if img is None:
        print(f"❌ Error: Image not found at {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    prediction = model.predict(img)
    return "Infected" if prediction[0] > 0.5 else "Uninfected"

# Test prediction
image_path = r"D:\neuralcare\malaria detection\cell_images\p7.png"  # Change to an actual image path
result = predict_image(image_path)
if result:
    print(f"✅ Prediction: {result}")
