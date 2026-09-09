import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def normal(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

def p(x):
    return 0.3*normal(x,2,1) + 0.4*normal(x,5,2) + 0.3*normal(x,9,1)

def sir(k, sample, q):
    x = sample(k)
    w = p(x) / q(x)
    w = w / w.sum()
    return np.random.choice(x, size=k, replace=True, p=w)

def plot(sample, q, name):
    xs = np.linspace(-5, 20, 1000)
    for k in [20, 100, 1000]:
        plt.figure()
        plt.hist(sir(k, sample, q), bins=np.arange(-5, 21, 1), density=True, alpha=0.6)
        plt.plot(xs, p(xs), "r")
        plt.title(f"{name}, k = {k}")
        plt.xlabel("Position x")
        plt.ylabel("Probability density")
        plt.show()

plot(lambda k: np.random.uniform(0, 15, k), lambda x: np.full_like(x, 1/15), "Uniform(0,15)")
plot(lambda k: np.random.normal(5, 4, k),   lambda x: normal(x, 5, 4),        "N(5,4)")