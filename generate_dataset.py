import numpy as np
import csv
from sklearn.datasets import make_classification, make_moons, load_breast_cancer

np.random.seed(42)

# Synthetic 1: Linearly separable
X1, y1 = make_classification(n_samples=1000, n_features=5, n_classes=2, n_informative=3, random_state=42)
with open("linear_separable.csv", "w", newline='') as file:
    writer = csv.writer(file)
    header = [f'x{i}' for i in range(X1.shape[1])] + ['y']
    writer.writerow(header)
    for i in range(X1.shape[0]):
        writer.writerow(list(X1[i]) + [y1[i]])

# Synthetic 2: Non-linear (moons)
X2, y2 = make_moons(n_samples=1000, noise=0.1, random_state=42)
with open("non_linear_moons.csv", "w", newline='') as file:
    writer = csv.writer(file)
    header = [f'x{i}' for i in range(X2.shape[1])] + ['y']
    writer.writerow(header)
    for i in range(X2.shape[0]):
        writer.writerow(list(X2[i]) + [y2[i]])

# Real-world: Breast cancer
data = load_breast_cancer()
X3, y3 = data.data, data.target
with open("breast_cancer.csv", "w", newline='') as file:
    writer = csv.writer(file)
    header = [f'x{i}' for i in range(X3.shape[1])] + ['y']
    writer.writerow(header)
    for i in range(X3.shape[0]):
        writer.writerow(list(X3[i]) + [y3[i]])