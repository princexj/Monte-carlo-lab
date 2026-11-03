import numpy as np
import matplotlib.pyplot as plt
from q1 import radical_inverse


def halton(n, base_x, base_y):
    pts = np.empty((n, 2))
    for i in range(1, n + 1):
        pts[i - 1, 0] = radical_inverse(i, base_x)
        pts[i - 1, 1] = radical_inverse(i, base_y)
    return pts


def scatter(pts, title, fname, size):
    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=size, color="darkgreen")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("first coordinate")
    plt.ylabel("second coordinate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


if __name__ == "__main__":
    for bx, by in ((2, 3), (97, 101)):
        for n in (100, 100000):
            pts = halton(n, bx, by)
            size = 14 if n == 100 else 0.5
            scatter(pts, f"Halton ({bx}, {by}), n = {n}", f"figs/q3_b{bx}_{by}_n{n}.png", size)
