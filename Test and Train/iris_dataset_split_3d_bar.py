import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into Train (60%), Validation (20%), Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Data for chart
split_sizes = [len(X_train), len(X_val), len(X_test)]
labels = ['Training (60%)', 'Validation (20%)', 'Testing (20%)']
colors = ['#4CAF50', '#FFC107', '#F44336']

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Coordinates for bars
x_pos = np.arange(len(split_sizes))
y_pos = np.zeros(len(split_sizes))  # All bars start at y = 0
z_pos = np.zeros(len(split_sizes))  # All bars start at z = 0

# Bar dimensions
dx = np.ones(len(split_sizes)) * 0.5  # width
dy = np.ones(len(split_sizes)) * 0.5  # depth
dz = split_sizes  # height

# Plot 3D bars
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

# Add labels
ax.set_xticks(x_pos + dx/2)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10, weight='bold')
ax.set_ylabel('Category', labelpad=15)
ax.set_zlabel('Number of Samples', labelpad=10)

# Title
ax.set_title('Dataset Split Distribution (3D)', fontsize=16, weight='bold')

# View angle
ax.view_init(elev=20, azim=45)

plt.show()
