import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

ticker = "AAPL"

# download adjusted closing prices
data = yf.download(ticker, period="max", auto_adjust=False, progress=False)
prices = data["Adj Close"].squeeze().dropna().values

# daily returns and annualized parameters
n = len(prices) - 1
returns = prices[1:] / prices[:-1] - 1
mu = 252 * returns.mean()
sigma = np.sqrt(252) * returns.std(ddof=1)
T = n / 252
s0 = prices[0]
actual = prices[-1]

# estimate S(T) using many simulated GBM terminal values
N = 100000
z = np.random.standard_normal(N)
sT = s0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
estimate = sT.mean()
std_err = sT.std(ddof=1) / np.sqrt(N)

# compare with the actual value
print("T =", T, "years")
print("S(0) =", s0)
print("mu =", mu)
print("sigma =", sigma)
print("Estimated S(T) =", estimate, "+/-", 1.96 * std_err)
print("Actual S(T) =", actual)
print("Relative error (%) =", (estimate - actual) / actual * 100)

# simulated GBM paths vs actual price path
dt = 1 / 252
t = np.arange(n + 1) * dt
steps = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal((10, n))
paths = s0 * np.exp(np.concatenate([np.zeros((10, 1)), np.cumsum(steps, axis=1)], axis=1))

plt.figure(figsize=(10, 5))
for path in paths:
    plt.plot(t, path, color="gray", lw=0.7)
plt.plot(t, s0 * np.exp(mu * t), "b", lw=2, label="expected path S(0)e^(mu t)")
plt.plot(t, prices, "r", lw=1.5, label="actual")
plt.yscale("log")
plt.xlabel("t (years)")
plt.ylabel("price")
plt.title(ticker + ": simulated GBM paths vs actual")
plt.legend()
plt.show()

# distribution of simulated S(T) with the actual value marked
plt.figure(figsize=(10, 5))
plt.hist(np.log(sT), bins=100, density=True, alpha=0.8)
plt.axvline(np.log(actual), color="r", lw=2, label="actual log S(T)")
plt.axvline(np.log(estimate), color="k", ls="--", label="log of estimated mean")
plt.xlabel("log S(T)")
plt.ylabel("density")
plt.title(ticker + ": distribution of simulated S(T)")
plt.legend()
plt.show()
