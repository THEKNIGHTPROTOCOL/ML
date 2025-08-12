# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Creating dummy array x (0 to 15), reshaped to 8x2
x = np.arange(16).reshape((8, 2))

# Creating dummy target variable y (0 to 7)
y = np.arange(8)

# Splitting data into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)

# Plotting the full dataset
plt.figure(figsize=(6, 4))
plt.plot(x)
plt.title("Full Dataset (Line Plot)")
plt.xlabel("Row index")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Scatter plot showing train vs test
plt.figure(figsize=(6, 4))
plt.scatter(x_train[:, 0], x_train[:, 1], color='blue', label='Train Data')
plt.scatter(x_test[:, 0], x_test[:, 1], color='red', label='Test Data')
plt.title("Train vs Test Split")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
