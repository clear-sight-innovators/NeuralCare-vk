import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define absolute dataset path
BASE_DIR = r"D:\neuralcare\malaria detection\cell_images"

def load_images(directory, label):
    images, labels = [], []
    
    if not os.path.exists(directory):
        print(f"❌ Error: Directory not found -> {directory}")
        return np.array([]), np.array([])
    
    for filename in os.listdir(directory):  
        img_path = os.path.join(directory, filename)

        # Skip non-image files
        if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            continue  # No need to print messages for skipping non-images
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {filename}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))  # Resize to model input size
        
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

def load_dataset():
    infected_dir = os.path.join(BASE_DIR, "Parasitized")
    uninfected_dir = os.path.join(BASE_DIR, "Uninfected")

    # Load images
    infected_images, infected_labels = load_images(infected_dir, 1)
    uninfected_images, uninfected_labels = load_images(uninfected_dir, 0)

    # Check if images were loaded
    if infected_images.size == 0 or uninfected_images.size == 0:
        print("❌ Error: No images loaded. Check dataset paths.")
        return None, None, None, None

    # Combine datasets
    X = np.concatenate((infected_images, uninfected_images), axis=0)
    y = np.concatenate((infected_labels, uninfected_labels), axis=0)

    # Normalize images
    X = X / 255.0  

    # Split dataset into training & test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

