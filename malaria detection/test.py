import tensorflow as tf
from utils import load_dataset

# Load dataset
X_train, X_test, y_train, y_test = load_dataset()

# Load trained model
model = tf.keras.models.load_model("malaria_model.h5")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
