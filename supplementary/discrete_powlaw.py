""" Sample from a discrete power law """
# Import necessary libraries
import numpy as np 
from scipy.special import zeta 

def draw_discrete(xmin, alpha, xmax=1000):
    x2 = xmin 
    x1 = xmax + 1
    Z0 = zeta(alpha, xmin)
    P = zeta(alpha, x2) / Z0    
    r = np.random.random() 
    # First determine the range of x
    while P >= 1-r:
        x1 = x2 
        x2 = 2*x1 
        P = zeta(alpha, x2) / Z0
    # Then pinpoint the solution by binary search    
    dx = abs(x2 - x1)
    while dx > 1:
        A = (zeta(alpha, x1) - zeta(alpha, x2)) / Z0
        if A > 1-r:
            x2 = (x2+x1) / 2
        else:
            x1 = (x2+x1) / 2
        dx = abs(x2 - x1)    
    return np.int64(x1)

def draw_discrete_approx(xmin, xmax, alpha):    
    C = -1 / (xmin**(1-alpha) - xmax**(1-alpha)) 
    # x = ((xmin-0.5)**(1-alpha) + np.random.random()/C)**(1/(1-alpha))
    x = xmax + 1
    while x > xmax:
        x = (xmin - 0.5)*(1-np.random.random())**(-1/(alpha-1)) + 0.5 
    return np.floor(x)


if __name__ == "__main__":
    np.random.seed(42)
    l0 = 1.
    alpha = 2.5

    N = 100000
    ell = np.zeros(N, dtype=np.int64)
    for n in range(N):
        ell[n] = draw_discrete_approx(l0, 1000, alpha)

    # Plot
    import matplotlib.pyplot as plt 
    bins = np.logspace(0, 3, 100).astype(np.int64)
    bins = np.unique(bins)
    hist, _ = np.histogram(ell, bins=bins, density=True)
    print(bins, hist)
    plt.loglog(bins[1:], hist, marker='o', linestyle='none')
    plt.show()