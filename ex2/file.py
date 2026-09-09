import random
import math
import matplotlib.pyplot as plt
 
k = 10000
 
def normal(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
 
 
def p(x):
    return (0.3 * normal(x, 2.0, 1.0)
            + 0.4 * normal(x, 5.0, 2.0)
            + 0.3 * normal(x, 9.0, 1.0))

 
def q(x):
    return 1 / 15
 

samples = []
for i in range(k):
    x = random.uniform(0, 50)
    samples.append(x)
 
w = []
for i in range(k):
    w.append(p(samples[i]) / q(samples[i]))
 
w_norm = []
for i in range(k):
    w_norm.append(w[i] / sum(w))
 
resampled = random.choices(samples, weights=w_norm, k=k)
 
grid = [i * 0.02 for i in range(-100, 801)]
curve = [p(x) for x in grid]
 
 
plt.hist(resampled, bins=30, range=(-2, 16), density=True,
         alpha=0.6, color="steelblue", edgecolor="white",
         label="resampled")
plt.plot(grid, curve, "r-", lw=2, label="p(x)")
plt.title("SIR, q = uniform(0, 15), k = " + str(k))
plt.xlabel("x")
plt.ylabel("density")
plt.legend()
plt.show()
 