# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into Train (60%), Temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Split Temp into Validation (20%) and Test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)



# 3D Pie Chart Code
split_sizes = [len(X_train), len(X_val), len(X_test)]
labels = ['Training\n(60%)', 'Validation\n(20%)', 'Testing\n(20%)']
colors = ['#4CAF50', '#FFC107', '#F44336']
explode = (0.1, 0.1, 0.1)



# Print sizes
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Visualization of split sizes
sizes = [len(X_train), len(X_val), len(X_test)]
labels = ['Train (60%)', 'Validation (20%)', 'Test (20%)']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Iris Dataset Split")
plt.axis('equal')
plt.show()
