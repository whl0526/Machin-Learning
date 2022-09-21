import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits(n_class=6)
one_idx = np.argwhere(digits.target == 1)
fig, ax = plt.subplots(5, 5, figsize=(6, 6))
j = 1

for i in range(int(one_idx.size)):
    if i in one_idx:
        plt.subplot(5, 5, j)
        plt.imshow(digits.images[i], cmap='binary')
        print(np.array(digits.images[i])[1:7, 2:6])
        j += 1
    if j > 25:
        break
plt.show()
