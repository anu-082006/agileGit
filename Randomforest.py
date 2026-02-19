import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sample training data
X = np.array([[50], [60], [70], [80], [90]])
y = np.array([55, 65, 75, 85, 95])

# Model
RandomForestRegModel = RandomForestRegressor(random_state=42)
RandomForestRegModel.fit(X, y)

# Prediction
X_marks = [[70]]
print(RandomForestRegModel.predict(X_marks))
