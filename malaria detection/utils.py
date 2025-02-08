import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(directory, label):
    images, labels = [], []
    
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        
        # Skip non-image files
        if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            print(f"Skipping non-image file: {filename}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {filename}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

def load_dataset():
    infected_dir = "cell_images/Parasitized"
    uninfected_dir = "cell_images/Uninfected"

    infected_images, infected_labels = load_images(infected_dir, 1)
    uninfected_images, uninfected_labels = load_images(uninfected_dir, 0)

    # Combine datasets
    X = np.concatenate((infected_images, uninfected_images), axis=0)
    y = np.concatenate((infected_labels, uninfected_labels), axis=0)

    # Normalize images
    X = X / 255.0  

    # Split dataset
    return train_test_split(X, y, test_size=0.2, random_state=42)
