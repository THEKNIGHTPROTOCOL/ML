import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# --- Load and Split Data ---
iris = load_iris()
X = iris.data
y = iris.target

# Train (60%), Validation (20%), Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Data
split_sizes = [len(X_train), len(X_val), len(X_test)]
labels = ['Training (60%)', 'Validation (20%)', 'Testing (20%)']
colors = ['#4CAF50', '#FFC107', '#F44336']

# Convert sizes to angles
total = sum(split_sizes)
angles = np.cumsum([0] + [s / total * 2 * np.pi for s in split_sizes])

# --- Create 3D Figure ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Height of pie chart
height = 0.3

# Draw 3D pie slices as cylindrical surfaces
for i in range(len(split_sizes)):
    theta = np.linspace(angles[i], angles[i+1], 30)
    r = 1
    T, Z = np.meshgrid(theta, [0, height])
    Xc = r * np.cos(T)
    Yc = r * np.sin(T)
    ax.plot_surface(Xc, Yc, Z, color=colors[i], alpha=0.9, shade=True)

    # Draw side edges for thickness
    x_side = r * np.cos(theta)
    y_side = r * np.sin(theta)
    ax.plot(x_side, y_side, 0, color='k', linewidth=0.5)
    ax.plot(x_side, y_side, height, color='k', linewidth=0.5)

    # Add labels above slices
    mid_angle = (angles[i] + angles[i+1]) / 2
    ax.text(1.2 * np.cos(mid_angle), 1.2 * np.sin(mid_angle), height / 2,
            f"{labels[i]}\n{split_sizes[i]} samples",
            ha='center', va='center', fontsize=10, weight='bold')

# Set equal scaling
ax.set_box_aspect([1, 1, 0.4])

# Remove axes for cleaner look
ax.set_axis_off()

# Title
ax.set_title('Dataset Split Distribution (True 3D Pie)', fontsize=16, weight='bold')

# Viewing angle
ax.view_init(elev=25, azim=45)

plt.show()
