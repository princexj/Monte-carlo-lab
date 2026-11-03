import numpy as np
import matplotlib.pyplot as plt
from q1 import van_der_corput


def lcg(n, seed=12345, a=1664525, c=1013904223, m=2**32):
    out = np.empty(n)
    state = seed
    for k in range(n):
        state = (a * state + c) % m
        out[k] = state / m
    return out


def compare(n, bins):
    vdc = van_der_corput(n)
    rnd = lcg(n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].hist(vdc, bins=bins, range=(0, 1), color="steelblue", edgecolor="white")
    axes[0].set_title(f"van der Corput, n = {n}")
    axes[1].hist(rnd, bins=bins, range=(0, 1), color="darkorange", edgecolor="white")
    axes[1].set_title(f"LCG (a=1664525, c=1013904223, m=2^32), n = {n}")
    for ax in axes:
        ax.set_xlabel("value")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("frequency")
    fig.tight_layout()
    fig.savefig(f"figs/q2_n{n}.png", dpi=200)
    plt.close(fig)

    expected = n / bins
    print(f"n = {n}, bins = {bins}, expected count per bin = {expected:.1f}")
    for name, data in (("van der Corput", vdc), ("LCG", rnd)):
        counts, _ = np.histogram(data, bins=bins, range=(0, 1))
        print(f"  {name:15s} min = {counts.min():6d}  max = {counts.max():6d}  max deviation = {np.abs(counts - expected).max():.1f}")


if __name__ == "__main__":
    compare(100, 20)
    compare(100000, 50)
