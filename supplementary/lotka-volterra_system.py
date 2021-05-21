""" Implements Lotka-Volterra systems for population dynamics on a lattice """
# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def immobile_LV(x0, t, lambda_, lambda_p, mu, sigma, K):
    a, b = x0 
    dXdt = [
        a*(lambda_*b - mu),
        sigma*b*(1 - (a+b)/K) - lambda_p*a*b
    ]
    return dXdt

if __name__ == "__main__":
    # Initialize
    lambda_ = 10
    lambda_p = 10
    mu = 0.25
    sigma = 0.5
    K = 1.5
    x0 = [0.5,0.5]
    T = 200
    t = np.linspace(0, T, 500*T)
    sol = odeint(immobile_LV, x0, t, args=(lambda_, lambda_p, mu, sigma, K))
    
    fig, ax = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
    # Plot a(t)
    ax.plot(t, sol[:,0], color='k', label="a(t)")
    a_star = (K-mu/lambda_)/(1+lambda_p*K/sigma)
    ax.plot([0,T],[a_star,a_star], color='k', linestyle='--', linewidth=0.5)
    # Plot b(t)
    ax.plot(t, sol[:,1], color='r', label="b(t)")
    ax.plot([0,T],[mu/lambda_, mu/lambda_], color='r', linestyle='--', linewidth=0.5)

    ax.legend()
    plt.show()