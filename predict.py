import joblib
import numpy as np
from sklearn.datasets import load_iris

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
data = load_iris()

sample = np.array([data.data[0]])   # shape (1,4)
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)
print("Predicted class index:", int(pred[0]), " name:", data.target_names[int(pred[0])])
