import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("tb_pneumonia_model.h5")

# Fix dataset path
TEST_DIR = r"D:/neuralcare/tb and pneumonia/dataset/test"

# Verify path
if not os.path.exists(TEST_DIR):
    print(f"ERROR: Test directory {TEST_DIR} does not exist!")
    exit(1)

# Image preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(150, 150), batch_size=32, class_mode='categorical'
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
