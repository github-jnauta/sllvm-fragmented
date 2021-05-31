""" Plot stuff """
# Import necessary libraries
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from matplotlib import rcParams
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
from scipy import integrate 
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import gmean, lognorm
from scipy.optimize import OptimizeWarning
# Set plotting font for TeX labels
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# Import modules 
import src.args 

# Set markers & colors
markers = ['s', '^', 'D', 'o', '>', '*', 'p', 'h', 'v', '<']
colors = [
    'firebrick', 'darkmagenta', 'darkgreen', 'navy', 'k', 'seagreen', 
    'darkorange', 'indigo', 'maroon', 'peru', 'orchid'
]
figlabels = [
    r'(a)', r'(b)', r'(c)', r'(d)', 
    r'(e)', r'(f)', r'(g)', r'(h)'
]
figbflabels = [
    r'\textbf{a}', r'\textbf{b}', r'\textbf{c}', r'\textbf{d}', 
    r'\textbf{e}', r'\textbf{f}', r'\textbf{g}', r'\textbf{h}'
]


class Plotter():
    def __init__(self):
        self.figdict = {}

    #########################
    # Lattice related plots #
    def plot_fragmented_lattice(self, args):
        _dir = args.ddir+"landscapes/"
        # Load lattice(s)
        _H = [0, 0.01, 0.05]
        L = 2**args.m
        # Initialize figure
        fig, axes = plt.subplots(1, len(_H), figsize=(3*len(_H), 3), tight_layout=True)
        for i, H in enumerate(_H):
            suffix = "_{L:d}x{L:d}_H{H:.3f}_rho{rho:.3f}".format(L=L, H=H, rho=args.rho)
            lattice = np.load(_dir+"lattice{suffix:s}.npy".format(suffix=suffix))
            axes[i].imshow(lattice, cmap='Greys')
        # Limits, labels, etc
        for i, ax in enumerate(axes):
            ax.set_xticks([0, L])
            ax.set_yticks([0, L])
            ax.set_xticklabels([r"0", r"L"], fontsize=14)
            ax.set_yticklabels([r"0", r"L"], fontsize=14)
            ax.xaxis.tick_top()
            ax.text(
                0.05, 0.9, r"H={:.2f}".format(_H[i]), transform=ax.transAxes,
                ha='left', fontsize=14, bbox=dict(boxstyle="round", ec='none', fc='white')
            )
        # Store
        self.figdict["example_lattices"] = fig 


    ## Static plots
    def plot_lattice_initial(self, args):
        # Specify directory
        _dir = args.ddir+"sllvm/{L:d}x{L:d}/".format(L=2**args.m)
        _rdir = "figures/"
        # Load lattice
        suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.3f}_lambda{:.3f}_sig{:.3f}_a{:.3f}".format(
            args.T, args.N0, args.M0, args.H, args.rho, args.mu, args.lambda_, args.sigma, args.alpha
        )
        lattice = np.load(_dir+"lattice{suffix:s}.npy".format(suffix=suffix))
        sites = np.load(_dir+"sites{suffix:s}.npy".format(suffix=suffix))
        # Reshape
        L_sq, _ = lattice.shape
        L = int(np.sqrt(L_sq))
        lattice = lattice.reshape(L,L,args.T)
        sites = sites.reshape(L,L)
        # Specify the colormap
        color_map = {
            -1: np.array([255, 0, 0]),      # prey, red
            0:  np.array([255, 255, 255]),  # empty, white
            1:  np.array([128, 128, 128]),  # eligible for prey, grey
            2:  np.array([0, 0, 0])         # predators, black
        }
        # Generate the image to be shown with correct colormap
        lattice[lattice>=1] = 2
        im = np.ndarray(shape=(L,L,args.T,3), dtype=np.int64)
        for i in range(0,L):
            for j in range(0,L):
                im[i,j,:,:] = color_map[sites[i,j]]
                for t in range(args.T):
                    if lattice[i,j,t]:
                        im[i,j,t,:] = color_map[lattice[i,j,t]] 
        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(5,5), tight_layout=True)
        ax.xaxis.tick_top()
        # Plot
        image = ax.imshow(im[:,:,0,:])
        # Store
        self.figdict["initial_lattice"] = fig


    ## Animated plots
    def plot_lattice_evolution(self, args):
        # Specify directory
        _dir = args.ddir+"sllvm/{L:d}x{L:d}/".format(L=2**args.m)
        _rdir = "figures/"
        # Set variables
        _alpha = [1, 2, 3]
        def get_image(alpha):
            # Load lattice
            suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}_lambda{:.3f}_sig{:.3f}_a{:.3f}_seed{:d}".format(
                args.T, args.N0, args.M0, args.H, args.rho, args.mu, args.lambda_, args.sigma, alpha, args.seed
            )
            lattice = np.load(_dir+"lattice{suffix:s}.npy".format(suffix=suffix))
            # pred_population = np.load(_dir+"pred_population{suffix:s}.npy".format(suffix=suffix))
            # print(pred_population)
            # sites = np.load(_dir+"sites{suffix:s}.npy".format(suffix=suffix))
            # Reshape
            L_sq, _ = lattice.shape
            L = int(np.sqrt(L_sq))
            lattice = lattice.reshape(L,L,args.nmeasures+1)
            # sites = sites.reshape(L,L)
            # Specify the colormap
            color_map = {
                -1: np.array([255, 0, 0]),      # prey, red
                0:  np.array([255, 255, 255]),  # empty, white
                1:  np.array([128, 128, 128]),  # eligible for prey, grey
                2:  np.array([0, 0, 0])         # predators, black
            }
            # Generate the image to be shown with correct colormap
            lattice[lattice>=1] = 2
            im = np.ndarray(shape=(L,L,args.nmeasures+1,3), dtype=np.int64)
            for i in range(0,L):
                for j in range(0,L):
                    # im[i,j,:,:] = color_map[sites[i,j]]
                    for t in range(args.nmeasures+1):
                        # if lattice[i,j,t]:
                        im[i,j,t,:] = color_map[lattice[i,j,t]]
            return im 
        # Initialize figure
        fig, axes = plt.subplots(1, 3, figsize=(3*len(_alpha),3), tight_layout=True)
        # Plot
        images = [get_image(alpha) for alpha in _alpha]
        ims = []
        for i, alpha in enumerate(_alpha):
            axes[i].xaxis.tick_top()
            axes[i].text(
                0.05, 0.9, r"$\alpha={:.2f}$".format(alpha), transform=axes[i].transAxes,
                fontsize=14, ha='left', bbox=dict(boxstyle="round", ec='none', fc='white')
            )
            ims.append(axes[i].imshow(images[i][:,:,0,:]))
        
        def update(t):
            for i, alpha in enumerate(_alpha):
                ims[i].set_array(images[i][:,:,t,:])
            return ims 

        anim = animation.FuncAnimation(fig, update, interval=25, frames=args.nmeasures+1)
        if not args.save:
            plt.show()
        else:
            suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.3f}_lambda{:.3f}_sig{:.3f}".format(
                args.T, args.N0, args.M0, args.H, args.rho, args.mu, args.lambda_, args.sigma
            )
            anim.save(
                _rdir+"gifs/lattice_animation{suffix:s}.gif".format(suffix=suffix),
                writer='imagemagick', fps=10
            )


    ## Static plots
    def plot_lattice(self, args):
        # Specify directory
        _dir = args.ddir+"sllvm/{L:d}x{L:d}/".format(L=2**args.m)
        # Load lattice
        suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.3f}_sig{:.3f}_a{:.3f}".format(
            args.T, args.N0, args.M0, args.H, args.rho, args.mu, args.sigma, args.alpha
        )
        lattice = np.load(_dir+"lattice{suffix:s}.npy".format(suffix=suffix))
        sites = np.load(_dir+"sites{suffix:s}.npy".format(suffix=suffix))
        L, _ = lattice.shape

        # Specify the colormap
        color_map = {
            -1: np.array([255, 0, 0]),      # prey, red
            0:  np.array([255, 255, 255]),  # empty, white
            1:  np.array([128, 128, 128]),  # eligible for prey, grey
            2:  np.array([0, 0, 0])         # predators, black
        }
        # Generate the image to be shown
        lattice[lattice>1] = 2
        img = np.ndarray(shape=(L,L,3), dtype=int)
        for i in range(0, L):
            for j in range(0, L):
                img[i,j] = color_map[sites[i,j]]            # Eligible sites
                if lattice[i,j] != 0:                       # Prey/predators
                    img[i,j] = color_map[lattice[i,j]]
        
        fig, ax = plt.subplots(1, 1, figsize=(6,6), tight_layout=True)
        ax.imshow(img, origin='lower')

    ############################
    # Population related plots #
    def plot_population_dynamics(self, args):
        L = 2**args.m
        _dir = args.ddir+"sllvm/{L:d}x{L:d}/".format(L=L)
        # Specify variables
        _alpha = [1, 2, 3]
        # Initialize figure
        fig, ax = plt.subplots(1,1, figsize=(6, 4), tight_layout=True)
        # Plot
        for i, alpha in enumerate(_alpha):
            suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}_lambda{:.4f}_sig{:.4f}_a{:.3f}_seed{:d}".format(
                args.T, args.N0, args.M0, args.H, args.rho, args.mu, args.lambda_, args.sigma, alpha, args.seed
            )
            _N = np.load(_dir+"pred_population{suffix:s}.npy".format(suffix=suffix)) 
            _M = np.load(_dir+"prey_population{suffix:s}.npy".format(suffix=suffix)) 
            N = np.mean(_N, axis=1) / L**2
            M = np.mean(_M, axis=1) / L**2
            # N = _N / L**2 
            # M = _M / L**2
            xax = args.T / args.nmeasures * np.arange(args.nmeasures+1)
            ax.plot(
                xax, N, color=colors[i], linestyle='-', linewidth=0.85
            )
            ax.plot(
                xax, M, color=colors[i], linestyle='--', linewidth=0.85, 
                label=r"$\alpha=%.1f$"%(alpha)
            )
        # Limits, labels, etc
        ax.set_xlim(0, args.T)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$t$", fontsize=14)
        ax.set_ylabel(r"population", fontsize=14)
        ax.legend(loc='upper right', fontsize=12, frameon=False)
    
    def plot_population_densities(self, args):
        L = 2**args.m 
        _rdir = args.rdir+"sllvm/{L:d}x{L:d}/".format(L=L)
        # Load variable arrays
        lambda_arr = np.loadtxt(_rdir+"lambda.txt")
        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(6,4), tight_layout=True)
        # Load data
        suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}_sig{:.4f}_a{:.3f}".format(
            args.T, args.N0, args.M0, args.H, args.rho, args.mu, args.sigma, args.alpha
        )
        _N = np.load(_rdir+f"N{suffix}.npy") / L**2
        _M = np.load(_rdir+f"M{suffix}.npy") / L**2
        N, M = np.mean(_N, axis=1), np.mean(_M, axis=1)        
        # Plot        
        ax_M = ax.twinx()
        ax.semilogx(
            lambda_arr, N, color='k', marker='o', mfc='white', markersize=4,
            label=r"$N(t)$"
        )
        ax_M.semilogx(
            lambda_arr, M, color='r', marker='s', mfc='white', markersize=4,
            label=r"$M(t)$"
        )
        # Limits, labels, etc
        ax.set_xlim(min(lambda_arr), max(lambda_arr))
        ax.set_ylim(bottom=0)
        ax_M.set_ylim(bottom=0)
        ax.set_xlabel(r"$t$", fontsize=14)
        ax.set_ylabel(r"population", fontsize=14)
        ax.legend(fontsize=13, loc='center left', frameon=False)
        

    ###########################
    # Levy walk related plots #
    def plot_predator_positions(self, args):
        _dir = args.ddir+"sllvm/{L:d}x{L:d}/".format(L=2**args.m)
        # Load positions
        suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.3f}_lambda{:.3f}_sig{:.3f}_a{:.3f}".format(
            args.T, args.N0, args.M0, args.H, args.rho, args.mu, args.lambda_, args.sigma, args.alpha
        )
        _x = np.load(_dir+"predator_positions{suffix:s}.npy".format(suffix=suffix))
        # Specify some other variables
        L = 2**args.m 
        # Compute 2D position for plotting
        N, T = _x.shape 
        x = np.zeros((N,2,T), dtype=np.int64)
        x[:,0,:] = _x // L 
        x[:,1,:] = _x % L 
        # Initialize figure
        fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)
        for n in range(N):
            ax.plot(
                x[n,0,:], x[n,1,:], linewidth=0.75
            )
        # Limits, labels, etc
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        
    

if __name__ == "__main__":
    Argus = src.args.Args() 
    args = Argus.args 
    Pjotr = Plotter()

    ## Lattice related plots
    # Pjotr.plot_lattice(args)
    # Pjotr.plot_predator_positions(args)
    # Pjotr.plot_lattice_evolution(args)
    # Pjotr.plot_lattice_initial(args)
    # Pjotr.plot_fragmented_lattice(args)
    ## Population density related plots
    # Pjotr.plot_population_dynamics(args)
    Pjotr.plot_population_densities(args)
    ## Dynamical system related plots
    
    if not args.save:
        plt.show()
    else:
        for figname, fig in Pjotr.figdict.items():
            print("Saving {}...".format(figname))
            fig.savefig("figures/{}.pdf".format(figname), bbox_inches='tight')