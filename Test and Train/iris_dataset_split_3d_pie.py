import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and split data
iris = load_iris()
X = iris.data
y = iris.target

# Split: Train (60%), Validation (20%), Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Data for pie chart
split_sizes = [len(X_train), len(X_val), len(X_test)]
labels = ['Training\n(60%)', 'Validation\n(20%)', 'Testing\n(20%)']
colors = ['#4CAF50', '#FFC107', '#F44336']
explode = (0.05, 0.05, 0.05)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))

# Fake "3D" by stacking pies
for z in np.linspace(0.08, 0, 8):  # depth layers
    wedges, _ = ax.pie(
        split_sizes, startangle=90, colors=colors,
        explode=explode, radius=1 - z, shadow=False
    )
    for w in wedges:
        w.set_alpha(0.6 if z > 0 else 1)

# Add top layer with labels
wedges, texts, autotexts = ax.pie(
    split_sizes, labels=labels, autopct='%1.1f%%',
    startangle=90, colors=colors, explode=explode,
    textprops={'fontsize': 12, 'weight': 'bold'}
)

plt.title('Dataset Split Distribution (3D-like Effect)', fontsize=14, weight='bold')
plt.show()
