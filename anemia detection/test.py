import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the test dataset
df_test = pd.read_csv(r"D:\neuralcare\anemia detection\anemia_test.csv")  # Provide test dataset path

# Check for missing values
df_test = df_test.dropna()

# Features and target
X_test = df_test[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']]
y_test = df_test['Result']

# Load the trained model
clf = joblib.load('anemia_model.pkl')

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.2f}")
