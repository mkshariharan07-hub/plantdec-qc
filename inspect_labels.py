import numpy as np
import os

if os.path.exists("labels.npy"):
    labels = np.load("labels.npy")
    print(sorted(list(set(labels))))
else:
    print("labels.npy not found")
