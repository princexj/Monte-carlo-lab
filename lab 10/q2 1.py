import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)

x0, mu, sigma = 5, 0.06, 0.3
n = 1000
dt = 1 / n
t = np.linspace(0, 1, n + 1)
lambdas = [0.01, 0.05, 0.1, 0.5, 1]

# (d) repeat everything for 5 sets of sample paths (the first set plus 4 more)
for s in range(5):
    dw = np.sqrt(dt) * np.random.standard_normal(n)
    diffusion = (mu - 0.5 * sigma ** 2) * dt + sigma * dw

    plt.figure(figsize=(10, 5))

    # (a) process with only the Wiener part, no jumps
    no_jump = x0 + np.concatenate(([0], np.cumsum(diffusion)))
    plt.plot(t, no_jump, "k", lw=2, label="no jumps")
    print(f"\nSet {s + 1}: X(1) without jumps = {no_jump[-1]:.4f}")

    # (b) process with Poisson jumps for each lambda
    for lam in lambdas:
        jumps = np.zeros(n)
        counts = np.random.poisson(lam * dt, n)
        for i in range(n):
            if counts[i] > 0:
                y = 0.1 * np.random.standard_normal(counts[i]) + 1
                jumps[i] = np.log(y).sum()
        path = x0 + np.concatenate(([0], np.cumsum(diffusion + jumps)))

        # (c) plot the jump path on the same graph as the no-jump path
        plt.plot(t, path, label=f"lambda = {lam}")
        print(f"  lambda = {lam}: jumps = {counts.sum()}, X(1) = {path[-1]:.4f}, max deviation from no-jump path = {np.max(np.abs(path - no_jump)):.4f}")

    plt.title(f"Set {s + 1}")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.legend()
    plt.show()
