""" Implements Lotka-Volterra systems for population dynamics on a lattice """
# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from scipy.integrate import odeint
from scipy.interpolate import interp1d
# Set plotting font for TeX labels
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def sigma(t, T=100):
    sigma = np.linspace(.5, 0.05, 500*T)
    return sigma[t]

def immobile_LV(x0, t, lambda_, lambda_p, mu, sigma, K):
    a, b = x0 
    dXdt = [
        a*(lambda_*b - mu),
        b*(sigma(t) - lambda_*a)
    ]
    return dXdt

def no_perturbation():
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
    ax.plot(t, sol[:,0], color='k', linewidth=1, label=r'foragers')
    # a_star = (K-mu/lambda_)/(1+lambda_p*K/sigma)
    # ax.plot([0,T],[a_star,a_star], color='k', linestyle='--', linewidth=0.5)
    # Plot b(t)
    ax.plot(t, sol[:,1], color=(88/235,196/235,221/235), linewidth=1, label=r'resources')
    # ax.plot([0,T],[mu/lambda_, mu/lambda_], color='r', linestyle='--', linewidth=0.5)
    # Limits, labels, etc
    ax.set_xlabel(r'time', fontsize=18)
    ax.set_ylabel(r'population', fontsize=18)
    ax.set_xlim(0,T)
    ax.set_ylim(0,25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc='upper right', labelspacing=0.2, fontsize=15, handlelength=1, borderaxespad=0, 
        handletextpad=0.4, frameon=False
    )
    fig.savefig('figures/lotka_volterra_example.pdf', bbox_inches='tight', transparent=True)
    plt.show()

def perturbation():
    """ Implements Lotka-Volterra systems for population dynamics on a lattice 
        and includes a perturbation that decreases prey populations at some time
    """
    # Initialize
    K = 1
    x0 = [7,10]
    T = 100
    t = np.linspace(0, T, 200*T)

    lambda_ = 0.075
    lambda_p = 1

    sample_times = np.array([0,t[-1]])

    # Mortality increases over time
    mu = np.array([.4,.8])
    fmu = interp1d(sample_times, mu, bounds_error=False, fill_value="extrapolate")
    # Prey reproduction decreases over time
    sigma = np.array([.6, .1])
    fsigma = interp1d(sample_times, sigma, bounds_error=False, fill_value="extrapolate")
    # Adaptation
    lambda_ = np.array([0.075,0.2])
    flambda = interp1d(sample_times, lambda_, bounds_error=False, fill_value="extrapolate")

    def LV(x0, t):
        a, b = x0
        dXdt = [
            a*(flambda(t)*b - fmu(t)),
            b*(fsigma(t) - flambda(t)*a)
        ]
        # if a < 1:
        #     dXdt[0] = -a
        return dXdt

    sol1 = odeint(LV, x0, t) + .5
    
    fig, ax = plt.subplots(1,1, figsize=(5,1.75), tight_layout=True)
    # Plot population dynamics    
    points = np.array([t,sol1[:,0]]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [(0,0,0), (62/252,0,0), (124/252,0,0), (186/252,0,0), (250/252,0,0,1)]
    cmap = LinearSegmentedColormap.from_list('mycmap', colors, gamma=.75)
    # cmap = ListedColormap(['r', 'g', 'b'])
    # norm = BoundaryNorm([0, .33, 0.66, 1], cmap.N)
    lc = LineCollection(segments, cmap=cmap)
    cols = np.linspace(0, 1, num=len(t))
    # cols = np.zeros(len(t))
    lc.set_array(cols)
    lc.set_linewidth(3)
    for i in range(3):
        line = ax.add_collection(lc)

    # ax.plot([0,T], [sol1[0,0],sol1[-1,0]], linestyle='--', color='k', linewidth=1.2)
    # ax.plot(t, sol1[:,0], color='darkviolet', linewidth=1.25, label=r'predator')
    # ax.plot(t, sol1[:,1], color='k', linestyle='-.', linewidth=1.25, label=r'prey')

    # ax.plot([t[tperturb],t[tperturb]], [0,25], color='k', linestyle=':', linewidth=.85)
    # Limits, labels, etc
    # ax.set_xlabel(r'time', fontsize=18)
    # ax.set_ylabel(r'population', fontsize=18)
    ax.set_xlim(0,T)
    ax.set_ylim(0,15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
    ax.plot(
        (1), (0), ls="", marker=">", ms=7, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False
    )
    # ax.legend(
    #     loc='upper right', labelspacing=0.2, fontsize=14, handlelength=1.5, borderaxespad=-.1, 
    #     handletextpad=0.4, frameon=False
    # )
    _dir = '/home/johannes/personal/postdoc/calls/WUR/proposal/'
    fig.savefig(_dir+'figures/population_dynamics.pdf', bbox_inches='tight', transparent=True)
    plt.show()

if __name__ == "__main__":
    perturbation()