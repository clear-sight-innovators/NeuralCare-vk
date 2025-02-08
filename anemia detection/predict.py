import joblib
import numpy as np

# Load trained model
clf = joblib.load('anemia_model.pkl')

# Take user input
gender = int(input("Enter Gender (0 for Male, 1 for Female): "))
hemoglobin = float(input("Enter Hemoglobin level: "))
mch = float(input("Enter MCH level: "))
mchc = float(input("Enter MCHC level: "))
mcv = float(input("Enter MCV level: "))

# Convert input into NumPy array
patient_data = np.array([[gender, hemoglobin, mch, mchc, mcv]])

# Predict anemia
prediction = clf.predict(patient_data)

if prediction[0] == 1:
    print("⚠️ The patient is likely Anemic.")
else:
    print("✅ The patient is NOT Anemic.")
