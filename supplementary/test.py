import numpy as np 
from scipy.special import zeta 
import numba 
import matplotlib.pyplot as plt 
from bisect import bisect_left
# Set plotting font for TeX labels
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def _F(k, alpha, xmin, xmax):    
    C = zeta(alpha, xmin) - zeta(alpha, xmax)
    return (zeta(alpha, k) - zeta(alpha, xmax))/C 

# @numba.jit(nopython=True, cache=True)
def sample_discrete_powlaw(N, alpha, xmin, xmax, F):
    """ Sample from trunctated discrete power law """
    samples = np.zeros(N, dtype=np.int64)
    K = len(F)
    for n in range(N):
        r = np.random.random() 
        idx = np.searchsorted(F[::-1], 1-r)
        samples[n] = xmin + K - idx - 1
    return samples 

## Specify constants
alpha = 1.1
N = 100000
xmin = 1
xmax = 10000
## Pre-compute Riemann zeta-function(s)
x = np.arange(xmin, xmax)
C = zeta(alpha, xmin) - zeta(alpha, xmax)
F = (zeta(alpha, x) - zeta(alpha, xmax))/C 
## Sample
np.random.seed(42)
samples = sample_discrete_powlaw(N, alpha, xmin, xmax, F)

## Plot
fig, ax = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
# Compute bins and histogram
bins = np.logspace(np.log10(xmin), np.log10(xmax), num=100, dtype=np.int64)
bins = np.unique(bins)
hist, _ = np.histogram(samples, bins=bins)
print(hist)
hist = hist / N 
cdf_sampled = np.cumsum(hist[::-1])[::-1]
# print(bins[:10], cdf_sampled[:10])
xplot = bins[:-1] #+ dbins

# Plot
ax.loglog(xplot, cdf_sampled, marker='o', linestyle='none', color='k', mfc='white', markersize=4)
ax.loglog(x, F, 'k--')
ax.set_xlabel(r'$\ell$', fontsize=16)
ax.set_ylabel(r'$P(x \geq \ell)$', fontsize=16)
plt.show()