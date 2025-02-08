import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and paths
IMG_SIZE = 64
parasitized_dir = "cell_images/Parasitized"
uninfected_dir = "cell_images/Uninfected"

# Function to load images
def load_images(directory, label):
    images = []
    labels = []
    valid_extensions = {".png", ".jpg", ".jpeg"}  # Allowed image types

    for file in os.listdir(directory):
        img_path = os.path.join(directory, file)
        ext = os.path.splitext(file)[-1].lower()
        
        if ext not in valid_extensions:  # Skip non-image files
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize
        images.append(img)
        labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

# Load and label the dataset
parasitized_images, parasitized_labels = load_images(parasitized_dir, 1)
uninfected_images, uninfected_labels = load_images(uninfected_dir, 0)

# Combine data
X = np.concatenate((parasitized_images, uninfected_images), axis=0)
y = np.concatenate((parasitized_labels, uninfected_labels), axis=0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Image augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model with augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=10)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# Function to predict malaria from an image
def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image.")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        print("Prediction: Parasitized (Infected)")
    else:
        print("Prediction: Uninfected")

# Example usage:
# predict_image("sample_image.jpg")
