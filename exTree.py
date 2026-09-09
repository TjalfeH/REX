import numpy as np
import matplotlib.pyplot as plt

# Normal destribition pdf
def norm_pdf(x, mean, std):
    return(
        1/ (np.sqrt(2* np.pi)*std)
        * np.exp(-0.5*((x-mean)/ std)**2)
    )

# Uniform proposal distribution q(x)
def props_dist(x):
    return np.ones_like(x) / 15.0

#Target sidtributing from the task
def targ_dist(x):
    return(
        0.3 * norm_pdf(x, 2.0, 1.0)
        + 0.4 * norm_pdf(x, 5.0, 2.0)
        + 0.3 * norm_pdf(x, 9.0, 1.0)
    )

# k- random x
def sir(K):

    make_sampel= np.random.uniform(0, 15, K)

    # p(x)/ p(q)
    weight=(
        targ_dist(make_sampel)
        / props_dist(make_sampel)
    )
    # Alle the ponct have a weight 
    weight = weight / np.sum(weight)


    # we resample agen between thoes weights, which they have hier 
    resam_higer= np.random.choice(
        make_sampel, 
        size =K,
        replace=True,
        p=weight
    )
    return resam_higer

# Histogram of the samples together with the wanted pose distribution p(x)
# It plots three histogram for 

x = np.linspace(0, 15, 1000)

for K in [20, 100, 1000]:
    result = sir(K)

    plt.figure()

    plt.hist(
        result,
        bins=20,
        density=True,
        alpha=0.6,
        label="Resampled samples"
    )

    plt.plot(
        x,
        targ_dist(x),
        color="red",
        label="p(x)"
    )

    plt.title(f"K = {K}")
    plt.xlabel("Position x")
    plt.ylabel("Probability density")
    plt.legend()
    plt.show()