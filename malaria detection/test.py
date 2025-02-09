import tensorflow as tf
from utils import load_dataset

# Load dataset from absolute path
DATASET_PATH = r"D:\neuralcare\malaria detection\cell_images\Parasitized"
X_train, X_test, y_train, y_test = load_dataset()

# Load trained model
model = tf.keras.models.load_model(r"D:\neuralcare\malaria detection\malaria_model.h5")

# Evaluate the model  
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")
