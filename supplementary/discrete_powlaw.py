""" Sample from a discrete power law """
# Import necessary libraries
import numpy as np 
from scipy.special import zeta 

def draw_discrete(xmin, alpha, N):
    x = np.zeros(N, dtype=np.int64)
    for n in range(N):
        x2 = xmin 
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
        while dx >= 1:
            P1 = zeta(alpha, x1) / Z0 
            P2 = zeta(alpha, x2) / Z0 
            if (P1+P2)/2 > 1-r:
                x1 = (x1+x2) / 2
            else:
                x2 = (x1+x2) / 2
            dx = abs(x2 - x1)
        x[n] = np.int64(x1)
    return x

def draw_discrete_approx(xmin, alpha, N):
    # x = ((xmin-0.5)**(1-alpha) + np.random.random()/C)**(1/(1-alpha))
    x = (xmin - 0.5)*(1-np.random.random(N))**(-1/(alpha-1)) + 0.5 
    return np.floor(x)

# @numba.jit(nopython=True, cache=True)
# def nb_sample_powlaw_discrete(alpha, C, xmin=1):
#     """ Generate discrete sample from the (truncated) powerlaw distribution
#         with normalization C (see https://www.jstor.org/stable/pdf/25662336.pdf)
#     """
#     _xmin = xmin - 0.5 
#     _sample = (_xmin**(1-alpha) + np.random.random()/C)**(1/(1-alpha))
#     _sample = np.floor(_sample)
#     return np.int64(_sample)


if __name__ == "__main__":
    np.random.seed(42)
    l0 = 1.
    alpha = 2.5

    N = 100000
    # ell = np.zeros(N, dtype=np.int64)
    # for n in range(N):
        # ell[n] = draw_discrete_approx(l0, alpha, N)
    ell_min = 1
    ell = draw_discrete(ell_min, alpha, N)
    ell_approx = draw_discrete_approx(ell_min, alpha, N)
    # Plot
    import matplotlib.pyplot as plt 
    # x = [5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 1000, np.max(ell)]
    x = np.logspace(np.log10(ell_min), np.log10(np.max(ell)), 30).astype(int)
    x = np.unique(x)
    PDF, _ = np.histogram(ell, bins=x, density=True)
    PDF_approx, _ = np.histogram(ell_approx, bins=x, density=True)
    CDF = np.zeros(len(x))
    CDF_approx = np.zeros(len(x))
    # Compute CDF immediately
    for i, x_ in enumerate(x):
        CDF[i] = np.sum(ell>=x_) / N 
        CDF_approx[i] = np.sum(ell_approx>=x_) / N
    # Compute error
    xmin = np.linspace(1, 20, num=20)
    error = 1 - ((xmin+0.5)/(xmin-0.5))**(1-alpha) - xmin**(-alpha)/zeta(alpha,xmin)
    # Initialize figure
    fig, axes = plt.subplots(1,3, figsize=(11.5, 3.5/4*3), tight_layout=True)
    # Plot empirical and true distribution
    # PDF
    xax = 10**(np.convolve(np.log10(x), np.ones(2, dtype=int), mode='valid')/2)
    axes[0].loglog(xax, PDF, marker='o', linestyle='none', markersize=4, mfc='none')
    axes[0].loglog(xax, PDF_approx, marker='s', linestyle='none', markersize=4, mfc='none')
    axes[0].loglog(x, x**(-alpha)/zeta(alpha,ell_min), color='k', linestyle='--')
    # CDF
    axes[1].loglog(x, CDF, marker='o', linestyle='none', markersize=4, mfc='none')
    axes[1].loglog(x, CDF_approx, marker='s', linestyle='none', markersize=4, mfc='none')
    axes[1].loglog(x, zeta(alpha,x)/zeta(alpha,ell_min), color='k', linestyle='--')
    # Plot error
    axes[2].plot(xmin, error, color='k')
    plt.show()