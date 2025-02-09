import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from model import build_model
from utils import load_dataset  # No need to pass arguments

# Load dataset
X_train, X_test, y_train, y_test = load_dataset()  # Call without arguments

# Build model  
model = build_model()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stop])

# Save trained model
model.save("malaria_model.h5")
print("✅ Model training complete. Saved as malaria_model.h5")
