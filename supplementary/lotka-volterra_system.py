""" Implements Lotka-Volterra systems for population dynamics on a lattice """
# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
# Set plotting font for TeX labels
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def immobile_LV(x0, t, lambda_, lambda_p, mu, sigma, K):
    a, b = x0 
    # dXdt = [
    #     a*(lambda_*b - mu),
    #     sigma*b*(1 - (a+b)/K) - lambda_p*a*b
    # ]
    dXdt = [
        a*(lambda_*b - mu),
        b*(sigma - lambda_*a)
    ]
    return dXdt

if __name__ == "__main__":
    # Initialize
    lambda_ = 0.1
    lambda_p = 1
    mu = 0.25
    sigma = 0.5
    K = 1
    x0 = [10,10]
    T = 100
    t = np.linspace(0, T, 500*T)
    sol = odeint(immobile_LV, x0, t, args=(lambda_, lambda_p, mu, sigma, K))
    
    fig, ax = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
    # Plot a(t)
    ax.plot(t, sol[:,0], color='k', linewidth=0.85, label=r'$x(t)$')
    # a_star = (K-mu/lambda_)/(1+lambda_p*K/sigma)
    # ax.plot([0,T],[a_star,a_star], color='k', linestyle='--', linewidth=0.5)
    # Plot b(t)
    ax.plot(t, sol[:,1], color='r', linewidth=0.85, label=r'$y(t)$')
    # ax.plot([0,T],[mu/lambda_, mu/lambda_], color='r', linestyle='--', linewidth=0.5)
    # Limits, labels, etc
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'population density', fontsize=15)
    ax.set_xlim(0,T)
    ax.set_ylim(0,25)
    ax.legend(
        loc='upper right', labelspacing=0.2, fontsize=14, handlelength=1, borderaxespad=0, 
        handletextpad=0.4, frameon=False
    )
    fig.savefig('figures/lotka_volterra_example.pdf', bbox_inches='tight')
    plt.show()