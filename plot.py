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
from scipy.special import zeta
# Set plotting font for TeX labels
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# Import modules 
import src.args 

# Set markers & colors
markers = ['s', 'o', 'D', '^', '>', '*', 'p', 'h', 'v', '<']
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

    #############
    # Functions #
    @staticmethod
    def true_diversity(N, M, q=1):
        if q == 1:
            pM = np.ma.divide(M,(M+N)).filled(0)
            pN = np.ma.divide(N,(M+N)).filled(0)
            return np.exp(- pM * np.ma.log(pM).filled(0) - pN * np.ma.log(pN).filled(0))
        else:
            basic_sum = N**q + M**q               
            return np.ma.power(basic_sum, (1/(1-q))).filled(0)

    #########################
    # Lattice related plots #
    def plot_fragmented_lattice(self, args):
        _dir = args.ddir+"landscapes/"
        # Load lattice(s)
        _H = [0.01, 0.1, 0.5, 0.99]
        _rho = [0.1, 0.2, 0.3, 0.4]
        L = 2**args.m
        # Initialize figure
        fig, axes = plt.subplots(1, len(_H), figsize=(2.5*len(_H), 2.5), tight_layout=True)
        # fig, axes = plt.subplots(1, len(_rho), figsize=(2.5*len(_rho), 2.5), tight_layout=True)
        for i, H in enumerate(_H):
        # for i, rho in enumerate(_rho):
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
                0.95, 0.1, r"H={:.2f}".format(_H[i]), transform=ax.transAxes,
                ha='right', fontsize=14, bbox=dict(boxstyle="round", ec='none', fc='white')
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

    def plot_patch_distribution(self, args):
        """ Plot measurements on the distribution over the patches generated 
            using fBm for different Hurst exponents H and occupancy levels ρ
        """
        # Specify some variables
        L = 2**args.m
        # Specify directory
        _dir = f'data/patch_distribution/{L}x{L}/'
        # Load variables
        H_arr = np.loadtxt(_dir+'H.txt')
        rho_arr = np.loadtxt(_dir+'rho.txt')
        # Specify bins for distribution plot
        bins = np.logspace(0, np.log10(0.1*L**2+1), num=args.nbins, dtype=np.int64)
        bins = np.unique(bins)
        # Initialize figure
        fig = plt.figure(figsize=(7, 7/4*3), tight_layout=True)
        gs = fig.add_gridspec(4,4)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        ax2 = fig.add_subplot(gs[0:2,2:4])
        ax3 = fig.add_subplot(gs[2:4,1:3])
        axes = [ax1, ax2, ax3]
        # Allocate
        patch_size = np.zeros((len(H_arr), args.nmeasures))
        num_patches = np.zeros((len(H_arr), args.nmeasures))
        
        # Load data & plot
        for i, rho in enumerate(rho_arr):
            for j, H in enumerate(H_arr):
                suffix = '_H{:.3f}_rho{:.3f}'.format(H, rho)
                patch_size[j,:] = np.load(_dir+f'patch_size{suffix}.npy')
                num_patches[j,:] = np.load(_dir+f'num_patches{suffix}.npy')
            # Plot
            mean_size = np.mean(patch_size, axis=1) / L**2 / rho
            axes[0].semilogx(
                H_arr, mean_size, color='k', marker=markers[i], mfc='white',
                markersize=4, label=r'$\rho=%.1f$'%(rho)
            )
            mean_num = np.mean(num_patches, axis=1)
            axes[1].semilogx(
                H_arr, mean_num, color='k', marker=markers[i], mfc='white',
                markersize=4, label=r'$\rho=%.1f$'%(rho)
            )

        for i, H in enumerate([0.01, 0.2, 0.5, 0.9]):
            # Plot distribution
            suffix = '_H{:.3f}_rho{:.3f}'.format(H, 0.1)
            pdf = np.load(_dir+f'patch_distribution{suffix}.npy')
            CCDF = np.cumsum(pdf[::-1])[::-1]
            color = matplotlib.colors.colorConverter.to_rgba('lightgrey')
            axes[2].loglog(
                bins/(L**2), CCDF, color='k', marker=markers[i], mfc=color, mec='k',
                markersize=4, label=r'$H=%.2f$'%(H), linestyle='--', dashes=(2,1)
            )
        axes[2].plot(
            [0.1,0.1], [0,1], color='k', linestyle=':', dashes=(2,2), linewidth=0.75
        )
        
        # Limits, labels, etc
        ylabels = [r'maximum patch size', r'mean number of patches', r'$P(X\geq x)$']
        ylims = [[0,1.05], [0,6e3], [1e-6,1.05]]
        for i, ax in enumerate(axes):
            xlim = [min(H_arr), 1] if i < 2 else [1/L**2,1]
            ax.set_xlim(xlim)                
            ax.set_ylim(ylims[i])
            if i < 2:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.set_xlabel(r'$H$', fontsize=16)
            else:
                ax.set_xlabel(r'$x/L^2$', fontsize=15)
                ax.text(
                    0.9, 0.55, r'$x=\rho L^2$', ha='center', transform=ax.transAxes,
                    fontsize=14, rotation=90
                )
                ax.legend(
                    loc='lower left', frameon=False, handletextpad=0.4, fontsize=13,
                    handlelength=1, borderaxespad=0.1, labelspacing=0.1
                )
                locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            fontsize = 14 if i < 2 else 16
            ax.set_ylabel(ylabels[i], fontsize=fontsize)
            if i == 1:
                ax.legend(
                    loc='center left', frameon=False, handletextpad=0.4, fontsize=13,
                    handlelength=1, borderaxespad=0.1, labelspacing=0.1
                )
            ax.text(
                0.0, 1.11, figlabels[i], ha='center', fontsize=15, transform=ax.transAxes
            )
        
        # Save
        self.figdict['patch_distribution'] = fig



    ## Animated plots
    def plot_lattice_evolution(self, args):
        # Specify directory
        _dir = args.ddir+"sllvm/evolution/{L:d}x{L:d}/".format(L=2**args.m)
        _rdir = "figures/"
        # Set variables
        def get_image(alpha):
            # Load lattice            
            suffix = (
                '_T{:d}_N{:d}_M{:d}_H{:.3f}'
                '_rho{:.3f}_mu{:.4f}_Lambda{:.4f}_lambda{:.4f}_sig{:.4f}_a{:.3f}'
                '_seed{:d}'.format(
                    args.T, args.N0, args.M0, args.H, args.rho, 
                    args.mu, args.Lambda_, args.lambda_, args.sigma, args.alpha,
                    args.seed
                )
            )
            lattice = np.load(_dir+"lattice{suffix:s}.npy".format(suffix=suffix))
            # Reshape
            L_sq, _ = lattice.shape
            L = int(np.sqrt(L_sq))
            lattice = lattice.reshape(L,L,args.nmeasures+1)
            # Generate the image to be shown with correct colormap
            # sites = np.load(_dir+"sites{suffix:s}.npy".format(suffix=suffix))
            # sites = sites.reshape(L,L)
            # lattice[lattice>1] = 2            
            # Specify the colormap
            # color_map = {
            #     -1: np.array([255, 0, 0]),      # prey, red
            #     0:  np.array([255, 255, 255]),  # empty, white
            #     1:  np.array([255, 255, 255]),  # eligible for prey, grey
            #     2:  np.array([0, 0, 0])         # predators, black
            # }
            # im = np.ndarray(shape=(L,L,args.nmeasures+1,3), dtype=np.int64)
            # for i in range(L):
            #     for j in range(L):
            #         im[i,j,:,:] = color_map[sites[i,j]]
            #         for t in range(args.nmeasures+1):
            #             if lattice[i,j,t]:
            #                 im[i,j,t,:] = color_map[lattice[i,j,t]]
            return lattice 

        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(4,4), tight_layout=True)
        # Generate colormap and normalization for coloring
        cmap = matplotlib.colors.ListedColormap(['red', 'white', 'grey', 'black'])
        bounds=[-1,0,1,2,3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # Plot
        image = get_image(args.alpha)
        ax.xaxis.tick_top()
        # im = ax.imshow(image[:,:,0,:])
        im = ax.imshow(image[:,:,0], cmap=cmap, norm=norm)
        # Define update function for animation
        def update(t):
            lattice_t = image[:,:,t]
            im.set_array(lattice_t)
            return im
        # Create the animation
        anim = animation.FuncAnimation(fig, update, interval=25, frames=args.nmeasures+1)
        # Show or save
        if not args.save:
            plt.show()
        else:
            suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}_lambda{:.4f}_sig{:.4f}".format(
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
        _dir = args.rdir+'sllvm/evolution/'
        _rdir = args.rdir+'sllvm/evolution/{L:d}x{L:d}/'.format(L=L)
        # Load variables
        H_arr = np.loadtxt(_dir+'H.txt')
        xax = args.T / args.nmeasures * np.arange(args.nmeasures+1)
        # Initialize figure
        fig, axes = plt.subplots(1,2, figsize=(7,3.5/4*3), tight_layout=True)
        axin = axes[0].inset_axes([0.58,0.62,0.35,0.35])
        # Plot
        for i, H in enumerate(H_arr):
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_alpha{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, H,
                args.rho, args.Lambda_, args.lambda_, args.alpha, args.mu, args.sigma
            )
            # Plot population density
            _N = np.load(_rdir+"N%s.npy"%(suffix)) 
            _M = np.load(_rdir+"M%s.npy"%(suffix)) 
            N = np.mean(_N, axis=1) / L**2
            M = np.mean(_M, axis=1) / L**2
            axes[0].plot(
                xax, N, color=colors[i], linewidth=0.85, label=r'$H=%.2f$'%(H)
            )
            axes[0].plot(
                xax, M, color=colors[i], linestyle='--', linewidth=0.85
            )
            # Plot predators on habitat
            _ph = np.load(_rdir+"ph%s.npy"%(suffix))
            ph = np.mean(_ph, axis=1)
            axin.plot(
                xax, ph, color=colors[i], linewidth=0.85
            )
            # Plot habitat efficiency
            _etah = np.load(_rdir+"etah%s.npy"%(suffix))
            etah = np.mean(_etah, axis=1)
            axes[1].plot(
                xax, etah, color=colors[i], linewidth=0.85, label=r'$H=%.2f$'%(H)
            )
        # Limits, labels, etc
        ylabels = [r"population", r"$\eta_h$"]
        for i, ax in enumerate(axes):
            ax.set_xlim(0, args.T)
            if i == 0:
                ax.set_ylim(0,0.25)
            else:
                ax.set_ylim(0,1.05)
                ax.legend(
                    loc='lower left', fontsize=14, handlelength=1, handletextpad=0.4,
                    borderaxespad=0.1, labelspacing=0.2, frameon=False
                )
            ax.set_xlabel(r"$t$", fontsize=16)
            ax.set_ylabel(ylabels[i], fontsize=16)
            ax.text(
                0.0, 1.05, figlabels[i], fontsize=15, ha='center', transform=ax.transAxes
            )
        axin.set_xlim(0, args.T)
        axin.set_ylim(0,1.05)
        # axin.set_xlabel(r"$t$", fontsize=12)
        axin.set_ylabel(r"$p_h$", fontsize=12)
        # Save
        self.figdict[f'population_dynamics_alpha{args.alpha}'] = fig 

    def plot_population_phase_space(self, args):
        L = 2**args.m 
        _dir = args.ddir+"sllvm/{L:d}x{L:d}/".format(L=L)
        # Specify variables
        # lambda_arr = [0.05]
        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(6,6), tight_layout=True)
        # Load data
        lambda_arr = [0.005, 0.02, 0.05, 0.1]
        for i, λ in enumerate(lambda_arr):
            suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}_lambda{:.4f}_sig{:.4f}_a{:.3f}_seed{:d}".format(
                args.T, args.N0, args.M0, args.H, args.rho, args.mu, λ, args.sigma, args.alpha, args.seed
            )
            _N = np.load(_dir+f"pred_population{suffix}.npy") / L**2
            _M = np.load(_dir+f"prey_population{suffix}.npy") / L**2
            N, M = np.mean(_N, axis=1), np.mean(_M, axis=1)
            # Plot
            ax.plot(N, M, color=colors[i], label=r"$\lambda=%.4f$"%(λ))
        # Limits, labels, etc
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$M$")
        ax.legend(loc='upper center')
    
    def plot_population_densities(self, args):
        L = 2**args.m 
        # Specify directories
        _rdir = args.rdir+"sllvm/evolution/{L:d}x{L:d}/".format(L=L)
        # Initialize figure
        fig, axes = plt.subplots(1, 2, figsize=(7,3.5), tight_layout=True)
        # axin = axes[0].inset_axes((0.5, 0.05, 0.4, 0.4))
        # Define time axis
        t = args.T / args.nmeasures * np.arange(args.nmeasures+1)
        # Load data
        lambda_arr = [0.013]
        for i, lambda_ in enumerate(lambda_arr):
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_alpha{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, args.H,
                args.rho, args.Lambda_, lambda_, args.alpha, args.mu, args.sigma
            )
            _N = np.load(_rdir+f'N{suffix}.npy') / (args.rho*L**2)
            _M = np.load(_rdir+f'M{suffix}.npy') / (args.rho*L**2)
            N, M = np.mean(_N, axis=1), np.mean(_M, axis=1)
            # Plot population densities
            axes[0].plot(t, N, color='k', linewidth=0.85, label=r'$N(t)$')
            axes[0].plot(t, M, color='navy', linewidth=0.85, label=r'$M(t)$')
            # Plot phase plot
            axes[1].plot(N, M, color='k', linewidth=0.85)
        # Limits, labels, etc
        axes[0].set_xlim(0, args.T)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlabel(r'$t$', fontsize=16)
        axes[0].set_ylabel(r'$N(t)$, $M(t)$', fontsize=16)
        axes[0].legend(loc='upper right', fontsize=15, frameon=False)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel(r'$N$', fontsize=16)
        axes[1].set_ylabel(r'$M$', fontsize=16)
        # Save
        self.figdict["population_densities"] = fig 

    def plot_population_densities_alpha(self, args):
        """ Plot quasistationary population densities as a function of α,
            for different Hurst exponents H (and interaction rates Λ)
        """
        L = 2**args.m
        # Specify directory
        _dir = args.rdir+'sllvm/alpha/'
        _rdir = args.rdir+'sllvm/alpha/{L:d}x{L:d}/'.format(L=L)
        # Load variables
        alpha_arr = np.loadtxt(_dir+'alpha.txt')
        H_arr = np.loadtxt(_dir+'H.txt')
        # Initialize figure
        fig, axes = plt.subplots(3,1, figsize=(5/4*3,7), tight_layout=True)
        # Load data & plot 
        for i, H in enumerate(H_arr):
            # Load data
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, H, 
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma
            )
            _N = np.load(_rdir+'N{:s}.npy'.format(suffix))
            _M = np.load(_rdir+'M{:s}.npy'.format(suffix))
            N = np.mean(_N, axis=1) / L**2
            M = np.mean(_M, axis=1) / L**2
            # Plot
            axes[0].plot(
                alpha_arr, N, color=colors[i], marker=markers[i], mfc='white',
                markersize=4, label=r'$H=%.2f$'%(H)
            )
            axes[1].plot(
                alpha_arr, M, color=colors[i], marker=markers[i], mfc='white',
                markersize=4, label=r'$H=%.2f$'%(H)
            )
            D = (Plotter.true_diversity(N, M)-1)*(N+M)
            # D = N/(N+M)
            axes[2].plot(
                alpha_arr, D, color=colors[i], marker=markers[i], mfc='white',
                markersize=4, label=r'$H=%.2f$'%(H)
            )
        # Limits, labels, etc
        ylabels = [r'$N$', r'$M$', r'richness']
        for i, ax in enumerate(axes):
            ax.set_xlim(1, 3)
            ax.set_xlabel(r'$\alpha$', fontsize=16)
            ax.set_ylabel(ylabels[i], fontsize=16)
            if i == 1:
                ax.legend(
                    loc='lower left', fontsize=11, ncol=1, labelspacing=0.1,
                    handletextpad=0.1, borderaxespad=0.1, handlelength=1,
                    columnspacing=0.6, frameon=False
                )
            if i == 2:
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(fr'$\rho={args.rho:.2f}$', fontsize=16)
         
        # Save
        self.figdict[f'population_density_alpha_rho{args.rho}'] = fig 
    
    def plot_population_densities_lambda(self, args):
        """ Plot population densities versus λ """
        L = 2**args.m
        # Specify directory
        _dir = args.rdir+'sllvm/lambda/'
        _rdir = args.rdir+'sllvm/lambda/{L:d}x{L:d}/'.format(L=L)
        # Load variables
        lambda_arr = np.logspace(-3,0,35)
        # Initialize figure
        fig, axes = plt.subplots(1,3, figsize=(3*3.5,3.5/4*3), tight_layout=True)
        # Load data & plot
            # Load data
        suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}' \
            'Lambda{:.4f}_alpha{:.3f}_mu{:.4f}_sigma{:.4f}'.format(
            args.T, args.N0, args.M0,
            args.rho, args.Lambda_, args.alpha, args.mu, args.sigma
        )
        _N = np.load(_rdir+'N{:s}.npy'.format(suffix))
        _M = np.load(_rdir+'M{:s}.npy'.format(suffix))
        N = np.mean(_N, axis=1) / L**2
        M = np.mean(_M, axis=1) / L**2
        D = Plotter.true_diversity(N, M)
        D = (N+M)*(D-1)
        # Plot
        axes[0].semilogx(
            lambda_arr, N, color='k', marker='o', mfc='white',
            markersize=4#, label=r'$H=%.2f$'%(H)#label=r'$\alpha=%.1f$'%(alpha)
        )
        axes[1].semilogx(
            lambda_arr, M, color='k', marker='o', mfc='white',
            markersize=4#, label=r'$H=%.2f$'%(H)#label=r'$\alpha=%.1f$'%(alpha)
        )
        axes[2].semilogx(
            lambda_arr, D, color='k', marker='o', mfc='white', markersize=4
        )
        # Locally compute a parabola to determine approximate maximum
        X = np.log10(lambda_arr)[7:16]
        Y = D[7:16]
        fit_coeffs = np.polyfit(X, Y, deg=2)
        fitD = np.poly1d(fit_coeffs)
        dfitD = fitD.deriv().r
        lambda_max = dfitD[dfitD.imag==0].real[0]
        axes[2].annotate(
            r'$\lambda^*\approx{:.3f}$'.format(10**lambda_max), xy=(10**lambda_max, fitD(lambda_max)),
            xytext=(0.0012, 0.315), ha='left', fontsize=13, 
            arrowprops=dict(fc='k', arrowstyle='->')
        )
        # Plot the line where λ=σ
        axes[2].semilogx(
            [args.sigma,args.sigma], [0,0.35], color='k', linestyle='--',
            linewidth=0.85
        )
        axes[2].text(
            1.15*args.sigma, 0.05, r'$\lambda=\sigma$', 
            rotation=0, ha='left', va='center', fontsize=13
        )
        # Limits, labels, etc
        ylabels = [r'$N$', r'$M$', r'$\mathcal{R}$']
        ylims = [0.12, 1.02, 0.35]
        height = [0.1, 0.1]
        for i, ax in enumerate(axes):
            ax.set_xlim(min(lambda_arr), max(lambda_arr))
            ax.set_ylim(0, ylims[i])
            ax.set_xlabel(r'$\lambda$', fontsize=16)
            ax.set_ylabel(ylabels[i], fontsize=16)
            ax.text(
                0.0, 1.065, figlabels[i], fontsize=14, ha='center', transform=ax.transAxes
            )
            # Plot some visual helpers
            ax.semilogx(
                [10**lambda_max, 10**lambda_max], [0,ax.get_ylim()[1]], 
                color='k', linestyle=':', linewidth=0.85
            )
            if i < 2:                
                _x, _y = (10**lambda_max)*(1+(-1)**(i)*0.15), ax.get_ylim()[1]*height[i]
                ha = 'right' if i%2 else 'left'
                ax.text(
                    _x, _y, r'$\lambda=\lambda^*$', fontsize=13, ha=ha,
                )
            if i == 0:
                ax.text(
                    0.05, 0.5, r'low reproduction', rotation=90, va='center', ha='left',
                    fontsize=11, transform=ax.transAxes
                )
                ax.text(
                    0.95, 0.5, r'overconsumption', rotation=90, va='center', ha='right',
                    fontsize=11, transform=ax.transAxes
                )
            # if i == 1:
            #     ax.legend(
            #         loc='upper right', fontsize=11, ncol=1, labelspacing=0.1,
            #         handletextpad=0.1, borderaxespad=0.1, handlelength=1,
            #         columnspacing=0.4, frameon=False
            #     )
        # Save
        self.figdict[f'population_densities_lambda_rho{args.rho}'] = fig 


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
    
    ################################
    # Flight lengthr related plots #
    def plot_flight_distribution_Lambda(self, args, xmin=1, xmax=None):
        """ Plot the distribution over flight lengths for different Λ """ 
        _dir = args.ddir+"sllvm/flights/{L:d}x{L:d}/".format(L=2**args.m)
        # Precompute some variables
        # Compute maximum flight length 
        xmax = 1*2**args.m if not xmax else xmax 
        xmax_measure = 1*2**args.m if not xmax else xmax 
        # Pre-compute the bins for distribution over flight lenghts
        bins = np.logspace(np.log10(xmin), np.log10(xmax_measure), num=args.nbins, dtype=np.int64)
        bins = np.unique(bins)
        locs = bins[:-1] + np.diff(np.log10(bins))
        ## Pre-compute Riemann zeta-function(s)
        x = np.arange(xmin, xmax)
        C = zeta(args.alpha, xmin) - zeta(args.alpha, xmax)
        F = (zeta(args.alpha, x) - zeta(args.alpha, xmax))/C 
        # Initialize figure
        fig, ax = plt.subplots(1,1, figsize=(5.25,3), tight_layout=True)
        # Load flight length distributions
        Lambda = np.loadtxt(_dir+"Lambda.txt")
        for i, Λ in enumerate(Lambda):
            suffix = (
                '_T{:d}_N{:d}_M{:d}_H{:.3f}'
                '_rho{:.3f}_mu{:.4f}_Lambda{:.4f}_lambda{:.4f}_sig{:.4f}_a{:.3f}'
                '_seed{:d}'.format(
                    args.T, args.N0, args.M0, args.H, args.rho, 
                    args.mu, Λ, args.lambda_, args.sigma, args.alpha, args.seed
                )
            )
            pdf = np.load(_dir+"flight_lengths%s.npy"%(suffix))
            pdf = pdf / np.sum(pdf)
            CCDF = np.cumsum(pdf[::-1])[::-1]
            ax.loglog(
                bins[:-1], CCDF, color=colors[i], label=r'$\Lambda=%.1e$'%(Λ),
                linestyle='none', marker=markers[i], mfc='none', markersize=3.5
            )
            ax.loglog(
                x, F, color='k', linestyle='--', linewidth=1
            )
        # Limits, labels, etc
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(top=1.1)
        ax.set_xlabel(r'$\ell$', fontsize=15)
        ax.set_ylabel(r'$P(\ell)$', fontsize=15)
        ax.legend(
            loc='lower left', fontsize=14, handlelength=1, handletextpad=0.1, 
            labelspacing=0.2, frameon=False
        )
        
    

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
    # Pjotr.plot_patch_distribution(args)

    ## Population density related plots
    # Pjotr.plot_population_dynamics(args)
    Pjotr.plot_population_densities(args)
    # Pjotr.plot_population_densities_alpha(args)
    # Pjotr.plot_population_densities_lambda(args)
    # Pjotr.plot_population_phase_space(args)

    ## Flight length related plots
    # Pjotr.plot_flight_distribution_Lambda(args)

    ## Dynamical system related plots
    
    if not args.save:
        plt.show()
    else:
        for figname, fig in Pjotr.figdict.items():
            print("Saving {}...".format(figname))
            fig.savefig("figures/{}.pdf".format(figname), bbox_inches='tight')