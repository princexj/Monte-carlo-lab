import numpy as np
import matplotlib.pyplot as plt


def radical_inverse(i, base):
    result = 0.0
    scale = 1.0 / base
    while i > 0:
        i, digit = divmod(i, base)
        result += digit * scale
        scale /= base
    return result


def van_der_corput(n, base=2):
    return np.array([radical_inverse(i, base) for i in range(1, n + 1)])


if __name__ == "__main__":
    first_25 = van_der_corput(25)
    print("First 25 values of the van der Corput sequence (base 2)")
    for i, x in enumerate(first_25, start=1):
        print(f"x_{i:<2d} = {x:.10f}")

    seq = van_der_corput(1000)
    plt.figure(figsize=(6, 6))
    plt.scatter(seq[:-1], seq[1:], s=8, color="navy")
    plt.xlabel(r"$x_i$")
    plt.ylabel(r"$x_{i+1}$")
    plt.title("Overlapping pairs $(x_i, x_{i+1})$, van der Corput, n = 1000")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/q1_pairs.png", dpi=200)
    plt.close()
