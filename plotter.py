import numpy as np
import matplotlib.pyplot as plt


def load_matrix(path):
    return np.loadtxt(path)


kernels = []
for i in range(8):
    A = load_matrix(
        f"/home/omar/CLionProjects/random-walks/cmake-build-debug/bin/kernel{i}.txt"
    )
    kernels.append(A)

global_min = min(A.min() for A in kernels)
global_max = max(A.max() for A in kernels)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i, A in enumerate(kernels):
    im = axes[i].imshow(
        A,
        origin="upper",
        aspect="auto",
        vmin=global_min,
        vmax=global_max,
        cmap="viridis"
    )
    axes[i].set_title(f"Kernel {i}")

plt.tight_layout()
plt.show()
