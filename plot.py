""" Plot stuff """
# Import necessary libraries
import enum
from os import MFD_ALLOW_SEALING
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from matplotlib import rcParams
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator
import colorsys


import warnings
from scipy import integrate
from scipy.ndimage.interpolation import rotate
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.stats import gmean, lognorm
from scipy.optimize import OptimizeWarning
from scipy.special import zeta, gamma
# Set plotting font for TeX labels
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}' r'\usepackage{amsmath}' r'\setlength{\jot}{-4pt}'
    # 'savefig.bbox': 'tight'
})

# Import modules 
import src.args 

# Set markers & colors
markers = ['s', 'o', 'D', 'p', '^', '*', '>', 'h', 'v', '<']
colors = [
    'k', 'navy', 'firebrick', 'darkgreen', 'darkmagenta',# 'seagreen', 
    'darkorange', 'indigo', 'maroon', 'peru', 'orchid'
]
lightcolors = [ 
    'lightgrey', 'steelblue', 'lightcoral', 'mediumseagreen', 'orchid', 
    'sandybrown', 'mediumpurple'
]
figlabels = [
    r'A', r'B', r'C', r'D', 
    r'E', r'F', r'G', r'H'
]
figbflabels = [
    r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', r'\textbf{D}', 
    r'\textbf{E}', r'\textbf{F}', r'\textbf{G}', r'\textbf{H}'
]
linestyles = ['-', '--', ':', '-.']

# set the colormap and centre the colorbar
class MidpointNormalize(matplotlib.colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

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
            pM = np.ma.divide(M,(M+N)).filled(0)
            pN = np.ma.divide(N,(M+N)).filled(0)
            basic_sum = pM**q + pN**q
            return np.ma.power(basic_sum, (1/(1-q))).filled(0)

    @staticmethod
    def fit_func(x, a, b, c):
        return a*x**3 * np.exp(-b*a*x) + c

    @staticmethod
    def sigmoid(x, a, b, c, d):
        return a * (1 / (1+np.exp(b*(x + c)))) + d
        
    @staticmethod
    def tanh(x, a, b, c, d):
        return a*np.tanh(b*x + c) + d
    
    @staticmethod
    def scale_lightness(rgb, scale_l):
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

    #########################
    # Lattice related plots #
    def plot_fragmented_lattice(self, args):
        _dir = args.ddir+"landscapes/"
        _pdfdir = args.ddir+'patch_distribution/{L:d}x{L:d}/'.format(L=2**args.m)
        # Load lattice(s)
        _H = [0.01, 0.2, 0.5, 0.99]
        _rho = [0.1, 0.2, 0.5, 0.9]
        L = 2**args.m
        # Initialize figure
        # fig, axes = plt.subplots(1,4, figsize=(4*1.525,3))
        fig, axes = plt.subplots(2,2, figsize=(3,3))
        figpdf, axpdf = plt.subplots(1,1, figsize=(2*4/3,2), tight_layout=True)
        _axes = axes.flatten()
        # Specify bins for distribution plot
        bins = np.logspace(0, np.log10(0.1*L**2+1), num=args.nbins, dtype=np.int64)
        bins = np.unique(bins)
        # fig, axes = plt.subplots(1, len(_rho), figsize=(2.5*len(_rho), 2.5), tight_layout=True)
        for i, H in enumerate(_H):
        # for i, rho in enumerate(_rho):
            suffix = "_{L:d}x{L:d}_H{H:.3f}_rho{rho:.3f}".format(L=L, H=H, rho=args.rho)
            lattice = np.load(_dir+"lattice{suffix:s}.npy".format(suffix=suffix))
            _axes[i].imshow(lattice, cmap='Greys', interpolation='none')
            pdfsuffix = '_H{:.3f}_rho{:.3f}'.format(H, args.rho)
            pdf = np.load(_pdfdir+f'patch_distribution{pdfsuffix}.npy')
            CCDF = np.cumsum(pdf[::-1])[::-1]
            axpdf.loglog(
                bins/(L**2), CCDF, color='k', marker=markers[i], mfc='white', mec='k',
                markersize=3, label=r'$H=%.2f$'%(H), linewidth=0.85, markevery=2
            )
        # Limits, labels, etc
        for i, ax in enumerate(_axes):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # r"$\rho={:.1f}$".format(_rho[i])
            # r"H={:.2f}".format(_H[i])
            ax.text(
                0.05, 0.925, rf'H={_H[i]:.2f}', transform=ax.transAxes, 
                ha='left', va='top', fontsize=12.5, 
                bbox=dict(boxstyle="round", ec='none', fc='white')
            )
            if i == 0:
                ax.text(
                    0.01, 1.01, figbflabels[i], ha='left', va='bottom',
                    fontsize=14, transform=ax.transAxes
                )
        # axpdf.set_xlim(1e-6, 1)
        # axpdf.set_ylim(1e-6,1.05)
        # locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        # axpdf.xaxis.set_minor_locator(locmin)
        # axpdf.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # axpdf.yaxis.set_minor_locator(locmin)
        # axpdf.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # axpdf.tick_params(axis='both', labelsize=9)
        # axpdf.set_xlabel(r'$x/L^2$', fontsize=14)
        # axpdf.set_ylabel(r'$P(X\geq x)$', fontsize=14)
        # axpdf.plot(
        #     [0.1,0.1], [0,1], color='k', linestyle='--', dashes=(2,2), linewidth=0.75, zorder=0
        # )
        # axpdf.text(
        #     0.915, 0.5, r'$x=\rho L^2$', ha='center', transform=axpdf.transAxes,
        #     fontsize=12, rotation=90
        # )
        # Store
        fig.subplots_adjust(wspace=0.03, hspace=0.03)
        self.figdict[f'example_lattices_rho{args.rho:.1f}'] = fig 
        # self.figdict[f'patch_size_distribution_rho{args.rho:.1f}'] = figpdf


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

    def plot_lattice_habitat_loss_static(self, args):
        """ Plot example of habitat loss for low and high levels of fragmentation """
        L = 2**args.m 
        # Specify directory
        _dir = args.ddir+f'sllvm/habitat_loss_example/{L}x{L}/'
        H_arr = [0.01, 0.50]
        # alphastar_R = [1.4727, 1.1957]
        alphastar_R = [2., 2.]
        # Initialize figure
        fig, axes = plt.subplots(2,3, figsize=(3,2))

        for i, H in enumerate(H_arr):
            __dir = _dir+f'H{H:.4f}/'
            suffix = (
            '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_mu{:.4f}_'
            'Lambda{:.4f}_lambda{:.4f}_sigma{:.4f}_alpha{:s}_seed{:d}'.format(
                args.T, args.N0, args.M0, H, args.rho, 
                args.mu, args.Lambda_, args.lambda_, args.sigma, '{alpha:.3f}',
                args.seed
                )
            )
            # Load & plot initial habitat
            _init_habitat = np.load(__dir+f'init_habitat{suffix.format(alpha=3.)}.npy')
            axes[i,0].imshow(_init_habitat, cmap='Greys', interpolation='none')
            # Load & plot for Brownian predators α=3
            M = np.load(__dir+f'pred_population{suffix.format(alpha=3)}.npy')
            _final_habitat = np.load(__dir+f'final_habitat{suffix.format(alpha=3.)}.npy')   
            axes[i,1].imshow(_final_habitat, cmap='Greys', interpolation='none')
            # Load & plot for optimal predators α*
            alpha = alphastar_R[i] if H == 0.01 else 2.
            _final_habitat = np.load(__dir+f'final_habitat{suffix.format(alpha=alpha)}.npy')
            axes[i,2].imshow(_final_habitat, cmap='Greys', interpolation='none')
            
        for i in range(len(H_arr)):
            axes[i,0].text(
                -0.05, 0.5, rf'$H={H_arr[i]:.2f}$', ha='right', va='center',
                rotation=90, fontsize=12, transform=axes[i,0].transAxes
            )
        # Limits, labels, etc
        alphalabs = ['', r'$\alpha=3.0$', r'$\alpha=1.47$', '', r'$\alpha=3.0$', r'$\alpha=2.0$']
        for i, ax in enumerate(axes.flatten()):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.text(
                    0.5, 1.01, r'$t=0$', fontsize=11, ha='center', va='bottom',
                    transform=ax.transAxes
                )
            if i == 1:
                ax.text(
                    1.05, 1.01, r'$t=T$', fontsize=11, ha='center', va='bottom',
                    transform=ax.transAxes
                )
            ax.text(
                0.5, -0.115, alphalabs[i], fontsize=8.5, ha='center', va='center',
                transform=ax.transAxes
            )
    
        fig.subplots_adjust(wspace=0.1, hspace=0.2)
        # Save
        self.figdict[f'example_lattice_habitat_loss_seed{args.seed:d}'] = fig 


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
        fig, ax = plt.subplots(1,1, figsize=(2.75,2.25/4*3))

        for i, H in enumerate([0.01, 0.2, 0.5, 0.9]):
            # Plot distribution
            suffix = '_H{:.3f}_rho{:.3f}'.format(H, 0.1)
            pdf = np.load(_dir+f'patch_distribution{suffix}.npy')
            CCDF = np.cumsum(pdf[::-1])[::-1]
            color = matplotlib.colors.colorConverter.to_rgba('lightgrey')
            ax.loglog(
                bins/(L**2), CCDF, color=colors[i], marker=markers[i], mfc='white',
                markersize=3.5, label=r'$H=%.2f$'%(H)
            )
        ax.plot(
            [0.1,0.1], [0,1], color='k', linestyle=':', dashes=(2,2), linewidth=0.75
        )
        
        # Limits, labels, etc        
        # for i, ax in enumerate(axes):
        ax.set_ylabel(r'$P(X\geq x)$', fontsize=14)
        ax.set_xlabel(r'$x/L^2$', fontsize=14)
        ax.set_xlim([1e-6,0.4])
        ax.set_ylim([1e-6,1.05])
        ax.text(
            0.9, 0.6, r'$x=\rho L^2$', ha='center', transform=ax.transAxes,
            fontsize=10, rotation=90
        )
        ax.legend(
            loc='lower left', frameon=False, handletextpad=0.2, fontsize=11,
            handlelength=1, borderaxespad=0, labelspacing=0.1
        )
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        xmin = 1/L**2
        ax.semilogx(
            [xmin,xmin], [1.5e-3,1.05], color='k', linestyle=':', linewidth=0.65
        )
        ax.text(
            xmin, 5e-3, r'$x=1$', rotation=90, ha='center', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='none', pad=0.1)
        )
        ax.set_xticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis='both', labelsize=9)
        
        # Save
        self.figdict['patchsize_distribution'] = fig

    def plot_patch_percolation(self, args):
        """ Plot percolation probability p as a function of ρ and H """
        L = 2**args.m
        # Specify variable arrays
        H_arr = [0.01, 0.2, 0.5, 1.0]
        rho_arr = [0.05*i for i in range(1,20)]
        xax = [0] + rho_arr + [1.]
        fitax = np.linspace(0, 1, num=250)
        # Specify directory
        _ddir = f'data/patch_distribution/{L:d}x{L:d}/'
        # Initialize figure        
        fig, axes = plt.subplots(1,2, figsize=(5,1.5))
        fig.subplots_adjust(wspace=0.35, hspace=0.075)
        # Load & plot
        for i, H in enumerate(H_arr):
            # Allocate
            xmax = np.zeros(len(rho_arr)+2)
            xmax[-1] = 1
            p = np.zeros(len(rho_arr)+2)
            p[-1] = 1.
            # Load data
            for j, rho in enumerate(rho_arr):
                suffix = '_H{:.3f}_rho{:.3f}'.format(H, rho)
                # Percolation probability
                _p = np.load(_ddir+f'percolation{suffix}.npy')
                p[j+1] = np.mean(_p)
                # Maximum patch size
                _xmax = np.load(_ddir+f'patch_size{suffix}.npy')
                xmax[j+1] = np.mean(_xmax) / (rho*L**2)
            # Plot xmax
            popt, _ = curve_fit(Plotter.sigmoid, xax[1:], xmax[1:])
            __fitax = fitax[np.argmax(fitax>=0.05):]
            axes[0].plot(__fitax, Plotter.sigmoid(__fitax, *popt), color=colors[i], linewidth=0.85)
            axes[0].plot(
                xax[1:], xmax[1:], color=colors[i], marker=markers[i], markersize=3.5, mfc='white', 
                linestyle='none'
            )
            axes[0].plot(
                [], [], color=colors[i], marker=markers[i], markersize=3.5, mfc='white', 
                linestyle='-', linewidth=0.85, label=rf'$H={H:.2f}$'
            )
            # Plot percolation
            popt, _ = curve_fit(Plotter.sigmoid, xax, p)
            axes[1].plot(fitax, Plotter.sigmoid(fitax, *popt), color=colors[i], linewidth=0.85)
            
            axes[1].plot(
                xax, p, color=colors[i], marker=markers[i], markersize=3.5, mfc='white', 
                linestyle='none'
            )
        # Limits, labels, etc
        ylabel = [r'$x_{\text{max}}/\rho L^2$', r'$p$']
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.set_xlabel(r'$\rho$', fontsize=16)
            ax.set_ylabel(ylabel[i], fontsize=16)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            if i == 0:
                ax.legend(
                    loc='lower right', fontsize=11.5, labelspacing=0., handlelength=1, 
                    handletextpad=0.1, borderaxespad=-0.1, frameon=False
                )
            else:
            
                ax.text(
                    0.18, 0.65, rf'$\rho={args.rho:.1f}$', rotation=90, ha='right', va='center',
                    fontsize=14
                )
            ax.text(
                0.01, 1.0, figbflabels[i+1], ha='left', va='bottom',
                fontsize=14, transform=ax.transAxes
            )
            # Some helper lines
            ax.plot(
                [args.rho, args.rho], [0, ax.get_ylim()[1]], color='k', linestyle=':', 
                linewidth=0.75, zorder=-1
            )
        # Save
        self.figdict[f'percolation'] = fig 
                


    ## Animated plots
    def plot_lattice_dynamics(self, args):
        # Specify directory
        _dir = args.ddir+"sllvm/dynamics/{L:d}x{L:d}/H{H:.4f}/".format(H=args.H,L=2**args.m)
        _rdir = "figures/"
        # Set variables
        def get_image(alpha):
            # Load lattice            
            suffix = (
                '_T{:d}_N{:d}_M{:d}_H{:.4f}'
                '_rho{:.3f}_mu{:.4f}_Lambda{:.4f}_lambda{:.4f}_sigma{:.4f}_alpha{:.3f}'
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
            suffix = (
                '_T{:d}_N{:d}_M{:d}_H{:.4f}'
                '_rho{:.3f}_mu{:.4f}_lambda{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
                    args.T, args.N0, args.M0, args.H, 
                    args.rho, args.mu, args.lambda_, args.sigma, args.alpha
                )
            )
            anim.save(
                _rdir+"gifs/lattice_animation{suffix:s}.gif".format(suffix=suffix),
                writer='imagemagick', fps=10
            )
    
    def plot_lattice_habitat_loss(self, args):
        """ Plot animated lattice evolution to illustrate habitat loss 
            Note: currently uses seed 42, as this seed (luckily) has some remaining
            prey habitat at T
        """
        L = 2**args.m 
        # Specify directory
        _dir = args.ddir+f'sllvm/habitat_loss_example/{L}x{L}/'
        H_arr = [0.01, 0.5]
        alphastar_R = [1.4727, 1.1957]

        # Initialize figures
        fig, axes = plt.subplots(2,2, figsize=(7,7))
        finfig, finaxes = plt.subplots(2,2, figsize=(5,5))
        # Define colormap and norm
        cmap = matplotlib.colors.ListedColormap(['black', 'white', 'lightgrey', 'red'])
        bounds=[-1,0,1,1.5,2.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        lattices = np.empty(axes.shape, dtype=object)
        images = np.empty(axes.shape, dtype=object)
        for i, H in enumerate(H_arr):
            __dir = _dir+f'H{H:.4f}/'
            suffix = (
                '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_mu{:.4f}_'
                'Lambda{:.4f}_lambda{:.4f}_sigma{:.4f}_alpha{:s}_seed{:d}'.format(
                    args.T, args.N0, args.M0, H, args.rho, 
                    args.mu, args.Lambda_, args.lambda_, args.sigma, '{alpha:.3f}',
                    args.seed
                )
            )
            # Load lattice for α scale-free
            _lattice = np.load(__dir+f'lattice{suffix.format(alpha=args.alpha)}.npy')
            __lattice = _lattice.reshape(L,L,args.nmeasures+1)
            lattices[i,0] = __lattice
            images[i,0] = axes[i,0].imshow(__lattice[:,:,0], cmap=cmap, norm=norm, interpolation='none')
            # Load lattice for α Brownian
            M = np.load(__dir+f'pred_population{suffix.format(alpha=3)}.npy')
            _lattice = np.load(__dir+f'lattice{suffix.format(alpha=3)}.npy')
            __lattice = _lattice.reshape(L,L,args.nmeasures+1)
            lattices[i,1] = __lattice
            images[i,1] = axes[i,1].imshow(__lattice[:,:,0], cmap=cmap, norm=norm, interpolation='none')

            # Plot final habitat for α scale-free
            _final_habitat = np.load(__dir+f'final_habitat{suffix.format(alpha=args.alpha)}.npy')
            finaxes[i,0].imshow(_final_habitat, cmap='Greys', interpolation='none')
            # Plot final habitat for α Brownian
            _final_habitat = np.load(__dir+f'final_habitat{suffix.format(alpha=3)}.npy')
            finaxes[i,1].imshow(_final_habitat, cmap='Greys', interpolation='none')
        
        for _axes in [axes, finaxes]:
            # Limits, labels, etc
            for i, ax in enumerate(_axes.flatten()):
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            # Manually add axis text
            _axes[0,0].text(
                -.05, 0.5, r'high fragmentation', rotation=90, color='red', 
                ha='right', va='center', fontsize=25, transform=_axes[0,0].transAxes
            )
            _axes[1,0].text(
                -.05, 0.5, r'low fragmentation', rotation=90, color='green', 
                ha='right', va='center', fontsize=25, transform=_axes[1,0].transAxes
            )
            _axes[0,0].text(
                0.5, 1.05, r'high dispersal rate', color='green', 
                ha='center', va='bottom', fontsize=25, transform=_axes[0,0].transAxes
            )
            _axes[0,1].text(
                0.5, 1.05, r'low dispersal rate', color='red', 
                ha='center', va='bottom', fontsize=25, transform=_axes[0,1].transAxes
            )

        # Animate evolution 
        def update(t):
            for i, H in enumerate(H_arr):
                # Update lattice for α scale free
                lattice_t1 = lattices[i,0][:,:,t]
                images[i,0].set_array(lattice_t1)
                # Update lattice for α Brownian
                lattice_t2 = lattices[i,1][:,:,t]
                images[i,1].set_array(lattice_t2)
        anim = animation.FuncAnimation(fig, update, interval=100, frames=args.nmeasures)
        
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        finfig.subplots_adjust(wspace=0.05, hspace=0.05)
        if not args.save:
            plt.show()
        else:
            anim.save(f'figures/gifs/habitat_loss_animation.gif', writer='imagemagick', fps=15)
            # writer = animation.FFMpegWriter(fps=15, bitrate=50000)
            # anim.save(
            #     f'figures/gifs/habitat_loss_animation.mp4', writer=writer
            # )
            self.figdict['habitat_loss_final'] = finfig

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
        # alpha_arr = np.loadtxt(_dir+'alpha.txt')
        alpha_arr = [1.1, 2.0, 3.0]
        H_arr = np.loadtxt(_dir+'H.txt')
        xax = args.T / args.nmeasures * np.arange(args.nmeasures+1)
        xax = xax / 1e4
        # Initialize figure
        fig, axes = plt.subplots(3,2, figsize=(2.5,4.5/4*3))
        figp, axesp = plt.subplots(1,3, figsize=(8,8/3), tight_layout=True)
        axesin = ['None'] + [ax.inset_axes([0.4,0.5,0.4*4/3,0.435]) for ax in axes.flatten()[1:]]
        axesin = np.array(axesin).reshape(3,2)        
        xaxin = xax[:len(xax)//10]
        # Set some stuff for the inset axes
        for i, ax in enumerate(axesin.flatten()[1:]):
            ax.set_xlim(0, xaxin[-1])
            ax.set_ylim(0.005, 0.195)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            # Reduce thickness of axis lines
            for _axis in ['top','bottom','left','right']:
                ax.spines[_axis].set_linewidth(0.5)
        # Plot
        _idx = -1 
        for i, alpha in enumerate(alpha_arr):
            for j, H in enumerate(H_arr):
                Hlab = rf'$H={H:.4f}$' if H == 0.9999 else rf'$H={H:.2f}$'
                suffix = suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_' \
                    'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.4f}'.format(
                    args.T, args.N0, args.M0, H,
                    args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma, alpha
                )
                # Plot population density
                _N = np.load(_rdir+"N%s.npy"%(suffix)) 
                _M = np.load(_rdir+"M%s.npy"%(suffix)) 
                N = np.mean(_N, axis=1) / L**2 
                M = np.mean(_M, axis=1) / L**2
                axes[i,0].plot(
                    xax, N, color=colors[j], linewidth=0.75, label=Hlab
                )
                axes[i,1].plot(
                    xax, M, color=colors[j], linewidth=0.75
                )
                if i > 0:
                    axesin[i,0].plot(
                        xaxin, N[:len(xaxin)], color=colors[j], linewidth=0.6
                    )
                axesin[i,1].plot(
                    xaxin, M[:len(xaxin)], color=colors[j], linewidth=0.6
                )
                axesp[i].plot(
                    N, M, color=colors[j], linewidth=0.85, label=Hlab
                )
        # Limits, labels, etc
        ylabels = [r'$N(t)$', r'$M(t)$']
        # alphalabels = [rf'$\alpha={:.1f}$', r'$\alpha=2.0$', r'$\alpha=3.0$']
        j = 0
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlim(0, args.T/1e4)
            ax.set_ylim(0,0.205)
            if i == 0:
                ax.legend(
                    loc='upper right', fontsize=7, handlelength=0.8, handletextpad=0.1,
                    borderaxespad=0., labelspacing=0.1, frameon=False
                )            
            ax.text(
                -0.1, 0.5, ylabels[i%2], fontsize=11, rotation=90, ha='right', va='center',
                transform=ax.transAxes
            )
            ax.text(
                0.01, 1.01, figbflabels[i], ha='left', va='bottom',
                fontsize=12, transform=ax.transAxes
            )
            if i % 2:
                ax.text(
                    1.1, 0.5, rf'$\alpha={alpha_arr[i//2]:.1f}$', fontsize=10, ha='center', 
                    va='center', rotation=90, transform=ax.transAxes
                )
                j += 1
            ax.set_yticks([0,0.2])
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
            ax.tick_params(axis='both', labelsize=7)
            if i > 3:
                ax.text(
                    0.5, -0.25, r'$t \; (\times 10^4)$', fontsize=10, ha='center', va='top',
                    transform=ax.transAxes
                )
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            else:
                ax.set_xticklabels([])
            # Reduce thickness of axis lines
            for _axis in ['top','bottom','left','right']:
                ax.spines[_axis].set_linewidth(0.5)
            # Indicate zoom axes
            if i > 0:
                rect, patches = ax.indicate_inset_zoom(
                    axesin.flatten()[i], edgecolor='k', linestyle='--', linewidth=0.5,
                    alpha=1
                )
                for _patch in patches:
                    _patch.set_linewidth(0.25)
                patches[0].set_visible(False)
                patches[1].set_visible(True)
                patches[2].set_visible(True)
                patches[3].set_visible(False)
        for i, ax in enumerate(axesin.flatten()[1:]):
            ax.set_ylim(0., 0.2)
            

        for i, ax in enumerate(axesp):
            ax.set_xlim(0, 1.05*args.rho)
            ax.set_ylim(0, 1.05*args.rho)
            ax.set_ylabel(r'$M$', fontsize=14)
            ax.text(
                0.5, 1.05, r'$\alpha=%.1f$'%(alpha_arr[i]), ha='center', fontsize=12,
                transform=ax.transAxes
            )
            if i == 2:
                ax.legend(
                    loc='center left', fontsize=12, frameon=False, labelspacing=0.1, handlelength=1.,
                    handletextpad=0.2, borderaxespad=0.2
                )
        # Save
        fig.subplots_adjust(wspace=0.5, hspace=0.3)
        self.figdict[f'population_dynamics_rho{args.rho:.2f}'] = fig 

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
        fig, axes = plt.subplots(2, 1, figsize=(3.25, 6), tight_layout=True)
        axin = axes[0].inset_axes((0.55, 0.45, 0.38, 0.38))
        # Define time axis
        t = args.T / args.nmeasures * np.arange(args.nmeasures+1)
        # Load data
        lambda_arr = [0.001, 0.013, 0.1]
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
            axes[0].plot(t, N, color=colors[i], linewidth=0.85, label=r'$\lambda=%.3f$'%(lambda_))
            axes[0].plot(t, M, color=colors[i], linestyle='--', linewidth=0.85)
            axin.plot(t, N, color=colors[i], linewidth=0.85)
            # Plot phase plot
            axes[1].plot(N, M, color=colors[i], linewidth=0.85, label=r'$\lambda=%.3f$'%(lambda_))
        # Limits, labels, etc
        axes[0].set_xlim(0, args.T)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlabel(r'$t$', fontsize=16)
        axes[0].set_ylabel(r'$N(t)$, $M(t)$', fontsize=16)
        axin.set_xlim(0, 2e3)
        axin.set_ylim(0,0.25)
        rect, patches = axes[0].indicate_inset_zoom(axin, ec='k', alpha=1, linewidth=0.5)
        for patch in patches:
            patch.set(linewidth=0.5, color='k', alpha=0.5)
        axes[1].legend(
            loc='upper right', fontsize=14, frameon=False, borderaxespad=0.2,
            handlelength=1, handletextpad=0.4, labelspacing=0.2
        )
        axes[1].set_xlim(0, 0.3)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel(r'$N$', fontsize=16)
        axes[1].set_ylabel(r'$M$', fontsize=16)
        for i, ax in enumerate(axes):
            ax.text(
                0.0, 1.05, figlabels[i], fontsize=14, ha='center', transform=ax.transAxes
            )
        # Save
        self.figdict[f'population_densities_rho{args.rho}'] = fig 

    def plot_population_densities_alpha(self, args):
        """ Plot quasistationary population densities as a function of α,
            for different Hurst exponents H (and interaction rates Λ)
        """
        L = 2**args.m
        # Specify directory        
        _dir = args.rdir+'sllvm/alpha/'
        _rdir = args.rdir+'sllvm/alpha/{L:d}x{L:d}/'.format(L=L)
        if args.compute:
            _rdir = _rdir+'fit/'
            _dir = _rdir
        
        # Load variables
        alpha_arr = np.loadtxt(_dir+'alpha.txt')
        idx_2 = np.argmax(alpha_arr==2.)
        # H_arr = np.loadtxt(_dir+'H.txt')
        H_arr = [0.01, 0.1, 0.2, 0.5, 1.0][::-1]
        # Initialize figure
        # fig, axes = plt.subplots(1,3, figsize=(8,2/4*3))
        fig, axes = plt.subplots(2,1, figsize=(2.5,4.5/4*3), sharex=True)
        axloss = axes[0].inset_axes([0.575,0.485,0.4,0.475])
        # figR, axR = plt.subplots(1,1, figsize=(3.5,3.5/4*3), tight_layout=True)
        # Load data & plot 
        _maxR = 0
        for i, H in enumerate(H_arr):
            # Load data
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, H, 
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma
            )
            _N = np.load(_rdir+'N{:s}.npy'.format(suffix)) / L**2
            _M = np.load(_rdir+'M{:s}.npy'.format(suffix)) / L**2
            N = np.mean(_N, axis=1)
            M = np.mean(_M, axis=1) 
            # Change label
            label = r'$H\rightarrow 1$' if H == 1 else r'$H=%.2f$'%(H)
            # Plot
            _i = len(H_arr) - i - 1    
            axes[0].plot(
                alpha_arr, N, color=colors[_i], marker=markers[_i], mfc='white',
                markersize=3.5, linewidth=0.85, label=label, zorder=len(H_arr)-_i
            )
            axes[1].plot(
                alpha_arr, M, color=colors[_i], marker=markers[_i], mfc='white',
                markersize=3.5, linewidth=0.85, label=label, zorder=len(H_arr)-_i
            )
            # _D = (Plotter.true_diversity(_N, _M, q=1)-1)
            # _R = _D * (_N+_M)
            # R = np.mean(_R, axis=1)
            # _maxR = max(_maxR, np.max(R))
            # axes[2].plot(
            #     alpha_arr, R, color=colors[_i], marker=markers[_i], mfc='white',
            #     markersize=3, linewidth=0.85, label=label, zorder=len(H_arr)-_i
            # )
            # Draw helper line to display catastropic extinction when predators
            # cannot adapt quickly to increased fragmentation
            if H == 1:
                Nhealthy = N 
                _alpha_max = alpha_arr[np.argmax(N==np.max(N))]
                # for ax in [axes[0], axes[2]]:
                axes[0].plot(
                    [_alpha_max, _alpha_max], [0,1], linestyle='--', linewidth=0.9,
                    color='k', zorder=-1
                )
            elif H < 1:
                __N = np.ma.divide(N, Nhealthy).filled(0)
                axloss.plot(
                    alpha_arr, __N, color=colors[_i], marker=markers[_i], mfc='white',
                    markersize=2.5, markevery=2, linewidth=0.75, zorder=len(H_arr)-_i
                )
        # Limits, labels, etc, for inset axes
        axloss.set_xlim(1,3.05)
        axloss.set_ylim(0,1.1)
        axloss.set_xticks([1,3])
        axloss.set_yticks([0,1])
        axloss.text(
            -0.125, 0.5, r'$N_{\text{rel}}$', fontsize=12, ha='right', va='center',
            transform=axloss.transAxes, rotation=90
        )
        axloss.text(
            0.5, -0.4, r'$\alpha$', fontsize=12, ha='center', va='center',
            transform=axloss.transAxes
        )
        axloss.xaxis.set_minor_locator(MultipleLocator(0.5))
        axloss.yaxis.set_minor_locator(MultipleLocator(0.25))
        axloss.tick_params(axis='both', which='major', labelsize=7)
        axloss.plot(
            [_alpha_max, _alpha_max], axloss.get_ylim(), linestyle='--', linewidth=1,
            color='k', zorder=-1
        )
        axloss.xaxis.set_minor_locator(MultipleLocator(0.25))
        axloss.xaxis.set_major_locator(MultipleLocator(1))
        # Limits, labels, etc, for remaining axes
        xlim = [1,2] if args.compute else [1,max(alpha_arr)]
        ylabels = [r'$N$', r'$M$', r'$\mathcal{R}$']
        ylims = [args.rho/2, 1.05*args.rho, 1.05*args.rho]        
        _axes = [ax for ax in axes]
        for i, ax in enumerate(_axes):
            if i == 0:
                ax.annotate(
                    r'$\alpha^*_{H\rightarrow 1}$', xy=(_alpha_max, 0.0925), xytext=(1.675,0.0925),
                    ha='center', va='center', arrowprops=dict(arrowstyle='->'), fontsize=10.5
                )     
            ax.set_xlim(xlim)
            ax.set_xticks([1., 1.5, 2., 2.5, 3.])
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.set_yticks([0,0.1,0.2])
            mloc = 0.025 if i == 0 else 0.05
            ax.yaxis.set_minor_locator(MultipleLocator(mloc))
            # ax.set_xticks([1+0.5*i for i in range(7)])
            ax.set_ylim(0, ylims[i])
            ax.set_ylabel(ylabels[i], fontsize=14)
            ax.text(
                0.01, 1., figbflabels[i], ha='left', va='bottom', fontsize=14,
                transform=ax.transAxes
            )
            if i == 1:
                ax.set_xlabel(r'$\alpha$', fontsize=14)
                ax.legend(
                    loc='upper right', fontsize=11, ncol=1, labelspacing=0.1,
                    handletextpad=0.1, borderaxespad=0.1, handlelength=1,
                    columnspacing=0.2, frameon=False
                )
                
        fig.subplots_adjust(wspace=0.03)
        # Save
        self.figdict[f'population_densities_alpha_rho{args.rho}'] = fig 
        # self.figdict[f'richness_alpha_rho{args.rho}'] = figR

    def plot_species_richness(self, args):
        """ Plot the species richness R """ 
        L = 2**args.m 
        ## Plot R as a function of α
        # Specify directory        
        _dir = args.rdir+'sllvm/alpha/'
        _rdir = args.rdir+'sllvm/alpha/{L:d}x{L:d}/'.format(L=L)
        # Load variables
        alpha_arr = np.loadtxt(_dir+'alpha.txt')
        H_arr = [0.01, 0.1, 0.2, 0.5, 1.0]
        # Initialize figure
        # fig, axes = plt.subplots(2,1, figsize=(2.75,5.25/4*3))
        fig, axes = plt.subplots(1, 2, figsize=(2*2.75,2.25/4*3), sharey=True)
        fig.subplots_adjust(hspace=0.3)
        for i, H in enumerate(H_arr):
            # Load data
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, H, 
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma
            )
            _N = np.load(_rdir+'N{:s}.npy'.format(suffix)) / L**2
            _M = np.load(_rdir+'M{:s}.npy'.format(suffix)) / L**2
            # Change label
            label = r'$H\rightarrow 1$' if H == 1 else r'$H=%.2f$'%(H)
            _D = (Plotter.true_diversity(_N, _M, q=1)-1)
            _R = _D * (_N+_M)
            R = np.mean(_R, axis=1)
            axes[0].plot(
                alpha_arr, R, color=colors[i], marker=markers[i], mfc='white',
                markersize=3.5, linewidth=0.85, label=label, zorder=len(H_arr)-i
            )
        ## PLot R as a function of H 
        # Specify directory
        _dir = args.rdir+'sllvm/H/'
        _rdir = args.rdir+'sllvm/H/{L:d}x{L:d}/'.format(L=L)
        alpha_arr = [1.5, 1.8, 2.0, 2.3, 2.5]
        H_arr = np.loadtxt(_dir+'H.txt')
        for i, alpha in enumerate(alpha_arr):
            suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
                args.T, args.N0, args.M0,
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma, alpha
            )
            _N = np.load(_rdir+f'N{suffix}.npy') / L**2
            _M = np.load(_rdir+f'M{suffix}.npy') / L**2
            _D = Plotter.true_diversity(_N, _M)
            _R = (_D-1)*(_N+_M)
            R = np.mean(_R, axis=1)
            axes[1].plot(
                H_arr, R, color=colors[i], marker=markers[i], mfc=lightcolors[i],
                markersize=3.5, linewidth=0.85, label=rf'$\alpha={alpha:.1f}$'
            )
        # Limits, labels, etc
        xlabels = [r'$\alpha$', r'$H$']
        xlims = [[1,3], [0,1]]
        ylim = [0.2, 0.2]
        for i, ax in enumerate(axes):
            if i == 0:
                ax.set_ylabel(r'$\mathcal{R}$', fontsize=14)
            ax.set_xlabel(xlabels[i], fontsize=14)
            ax.set_yticks([0.,0.1, 0.2])
            ax.set_xlim(xlims[i])
            ax.set_ylim(0, ylim[i])
            ax.text(
                0.0, 1., figbflabels[i], ha='left', va='bottom', fontsize=14,
                transform=ax.transAxes
            )
            if i == 0:
                ax.legend(
                    loc='upper right', fontsize=11, ncol=1, labelspacing=0.1,
                    handletextpad=0.1, borderaxespad=0.1, handlelength=1,
                    columnspacing=0.2, frameon=False
                )
            else:                
                (handles, labels) = plt.gca().get_legend_handles_labels()
                handles.insert(2, plt.Line2D([],[], linestyle='none'))
                labels.insert(2, '')
                ax.legend(
                    handles, labels, loc='upper right', fontsize=11, 
                    frameon=False, labelspacing=0.,
                    handletextpad=0.2, handlelength=0.9, borderaxespad=-0.1,
                    ncol=2, columnspacing=0.2
                )
            xloc = 0.1 if i == 0 else 0.05
            ax.xaxis.set_minor_locator(MultipleLocator(xloc))
            ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.15, hspace=0.5)
        # Save
        self.figdict[f'species_richness_rho{args.rho:.2f}'] = fig

    def plot_population_densities_lambda(self, args):
        """ Plot population densities versus λ """
        L = 2**args.m
        # Specify directory
        _dir = args.rdir+'sllvm/lambda/'
        _rdir = args.rdir+'sllvm/lambda/{L:d}x{L:d}/'.format(L=L)
        # Load variables
        lambda_arr = np.logspace(-3,0,25)
        alpha_arr = np.loadtxt(_dir+'alpha.txt')
        H_arr = np.loadtxt(_dir+'H.txt')
        # Initialize figure
        fig, axes = plt.subplots(1,3, figsize=(3*3.5,3.5/4*3), tight_layout=True)
        for i, alpha in enumerate(alpha_arr):
        # for i, H in enumerate(H_arr):
            # Load data
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}' \
                    '_Lambda{:.4f}_alpha{:.3f}_mu{:.4f}_sigma{:.4f}'.format(
                    args.T, args.N0, args.M0, args.H,
                    args.rho, args.Lambda_, alpha, args.mu, args.sigma
            )
            _N = np.load(_rdir+'N{:s}.npy'.format(suffix))
            _M = np.load(_rdir+'M{:s}.npy'.format(suffix))
            N = np.mean(_N, axis=1) / L**2
            M = np.mean(_M, axis=1) / L**2
            D = Plotter.true_diversity(N, M)
            D = (N+M)*(D-1)
            # Plot
            axes[0].semilogx(
                lambda_arr, N, color=colors[i], marker=markers[i], mfc='white',
                markersize=4, label=r'$\alpha=%.1f$'%(alpha) #label=r'$H=%.2f$'%(H)# 
            )
            axes[1].semilogx(
                lambda_arr, M, color=colors[i], marker=markers[i], mfc='white',
                markersize=4#, label=r'$\alpha=%.1f$'%(alpha) #label=r'$H=%.2f$'%(H)#
            )
            axes[2].semilogx(
                lambda_arr, D, color=colors[i], marker=markers[i], mfc='white', markersize=4
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
        ylims = [0.4, 0.21, 0.3]
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
                ax.legend(
                    loc='upper left', fontsize=11, ncol=1, labelspacing=0.1,
                    handletextpad=0.1, borderaxespad=0.1, handlelength=1,
                    columnspacing=0.4, frameon=False
                )
        # Save
        self.figdict[f'population_densities_lambda_rho{args.rho}'] = fig 

    def plot_population_densities_sigma(self, args):
        """ Plot population densities as a function of σ for several α and H """
        L = 2**args.m
        # Specify directory
        _dir = args.rdir+'sllvm/sigma/'
        _rdir = args.rdir+'sllvm/sigma/{L:d}x{L:d}/'.format(L=L)
        # Load variables
        # alpha_arr = np.loadtxt(_dir+'alpha.txt')
        alpha_arr = [3.0]
        sigma_arr = np.loadtxt(_dir+'sigma.txt')
        H_arr = np.loadtxt(_dir+'H.txt')
        # Initialize figure
        fig, axes = plt.subplots(1,2, figsize=(2*3.5,3.5/4*3), tight_layout=True)
        figR, axR = plt.subplots(1,1, figsize=(3.5,3.5/4*3), tight_layout=True)
        # Load & plot
        for i, H in enumerate(H_arr):
            for j, alpha in enumerate(alpha_arr):
                suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}' \
                    '_Lambda{:.4f}_mu{:.4f}_alpha{:.3f}'.format(
                    args.T, args.N0, args.M0, H,
                    args.rho, args.Lambda_, args.mu, alpha
                )
                # Load data
                _N = np.load(_rdir+f'N{suffix}.npy') / L**2
                _M = np.load(_rdir+f'M{suffix}.npy') / L**2
                N = np.mean(_N, axis=1)
                M = np.mean(_M, axis=1)
                _D = (Plotter.true_diversity(_N, _M)-1)
                _R = _D * (_N+_M)
                R = np.mean(_R, axis=1)
                # Plot
                axes[0].plot(
                    sigma_arr, N, color=colors[i], marker=markers[j], mfc='white',
                    markersize=3.5, linewidth=0.85, label=rf'$H={H:.2f}, \alpha={alpha:.2f}$'
                )
                axes[1].plot(
                    sigma_arr, M, color=colors[i], marker=markers[j], mfc='white',
                    markersize=3.5, linewidth=0.85, label=rf'$H={H:.2f}, \alpha={alpha:.2f}$'
                )
                axR.plot(
                    sigma_arr, R, color=colors[i], marker=markers[j], mfc='white',
                    markersize=3.5, linewidth=0.85, label=rf'$H={H:.2f}, \alpha={alpha:.2f}$'
                )
        
        # Limits, labels, etc
        for i, ax in enumerate(axes):
            ax.set_xlabel(r'$\sigma$', fontsize=16)
            if i == 0:
                ax.legend(
                    loc='upper right', fontsize=12, frameon=False, labelspacing=0.1,
                    handletextpad=0.1, handlelength=1, borderaxespad=0
                )

    def plot_population_densities_H(self, args):
        """ Plot populations densites as a function of H 
            Note that some of the plots are a function of α for different values of H
        """
        L = 2**args.m
        # Specify directory
        _dir = args.rdir+'sllvm/H/'
        _rdir = args.rdir+'sllvm/H/{L:d}x{L:d}/'.format(L=L)
        # Load variables
        # alpha_arr = np.loadtxt(_dir+'alpha.txt')
        alpha_arr = [1.5, 1.8, 2.0, 2.3, 2.5]
        H_arr = np.loadtxt(_dir+'H.txt')
        # Initialize figure        
        fig, axes = plt.subplots(2,1, figsize=(2.5,4.5/4*3), sharex=True)
        # Load data & plot 
        for i, alpha in enumerate(alpha_arr):
            suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
                args.T, args.N0, args.M0,
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma, alpha
            )
            _N = np.load(_rdir+f'N{suffix}.npy') / L**2
            _M = np.load(_rdir+f'M{suffix}.npy') / L**2
            N = np.mean(_N, axis=1) 
            M = np.mean(_M, axis=1) 
            print(alpha, N)
            axes[0].plot(
                H_arr, N,
                color=colors[i], marker=markers[i], mfc=lightcolors[i],
                markersize=3, linewidth=0.85, label=rf'$\alpha={alpha:.1f}$'
            )
            axes[1].plot(
                H_arr, M, color=colors[i], marker=markers[i], mfc=lightcolors[i],
                markersize=3, linewidth=0.85, label=rf'$\alpha={alpha:.1f}$'
            )
        # Limits, labels, etc
        ylabels = [r'$N$', r'$M$', r'$\mathcal{R}$']
        ymax = [0.1, 0.1, 0.2]
        for i, ax in enumerate(axes):
            ax.set_xticks([0,0.2,0.4,0.6,0.8,1])            
            ax.set_yticks([0, 0.1, 0.2])
            ax.yaxis.set_minor_locator(MultipleLocator(0.025))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, ymax[i])
            ax.set_ylabel(ylabels[i], fontsize=14)
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            if i == 1:
                ax.set_xlabel(r'$H$', fontsize=14)
                (handles, labels) = plt.gca().get_legend_handles_labels()
                handles.insert(2, plt.Line2D([],[], linestyle='none'))
                labels.insert(2, '')
                ax.legend(
                    handles, labels, loc='upper right', fontsize=11, 
                    frameon=False, labelspacing=0.,
                    handletextpad=0.2, handlelength=0.9, borderaxespad=-0.1,
                    ncol=2, columnspacing=0.2
                )
            ax.text(
                0.01, 1.0, figbflabels[i+2], ha='left', va='bottom', fontsize=14,
                transform=ax.transAxes
            )

        fig.subplots_adjust(wspace=0.03)
        # Save
        self.figdict[f'population_densities_H_rho{args.rho:.1f}'] = fig
        # self.figdict[f'richness_H_rho{args.rho:.1f}'] = figR

    def plot_population_densities_fragile(self, args):
        """ Plot population densities as a function of λ, but for different σ,
            as to determine for what value of λ the system is (most) fragile 
        """
        L = 2**args.m
        # Specify directory
        _dir = args.rdir+'sllvm/lambda/'
        _rdir = _dir+'{L:d}x{L:d}/'.format(L=L)
        # Load variables
        # alpha_arr = np.loadtxt(_dir+'alpha.txt')[::3]
        alpha_arr = [1.01, -1]
        # print(alpha_arr)
        sigma_arr = np.loadtxt(_dir+'sigma.txt')
        lambda_arr = np.loadtxt(_dir+'lambda.txt')
        lambda_star = np.zeros(len(sigma_arr))
        # Initialize figure
        fig, axes = plt.subplots(2,1, figsize=(3,5/4*3), sharex=True)
        # fig, axes = plt.subplots(1, 2, figsize=(2*3.25,2.25/4*3))
        fig.subplots_adjust(wspace=0.275)
        axin = axes[0].inset_axes((0.16, 0.45, 0.45, 0.45))
        # Load and plot
        for i, sigma in enumerate(sigma_arr):            
            for j, alpha in enumerate(alpha_arr):
                suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}' \
                    '_Lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
                    args.T, args.N0, args.M0, args.H,
                    args.rho, args.Lambda_, args.mu, sigma, alpha
                )
                _N = np.load(_rdir+f'N{suffix}.npy') / L**2
                _M = np.load(_rdir+f'M{suffix}.npy') / L**2
                N = np.mean(_N, axis=1)
                M = np.mean(_M, axis=1)
                # R = (N+M)*(Plotter.true_diversity(N,M) - 1)
                opacity = 0.8 if alpha==-1 else 1
                mfc = matplotlib.colors.to_rgba('white', opacity)
                line, = axes[0].semilogx(
                    lambda_arr, N, color=colors[i], marker=markers[i], linestyle=linestyles[j],
                    linewidth=0.85, mfc=mfc, markersize=3.5, label=rf'$\sigma={sigma:.2f}$'
                )
                axes[1].semilogx(
                    lambda_arr, M, color=colors[i], marker=markers[i], linestyle=linestyles[j],
                    linewidth=0.85, mfc='white', markersize=3.5, label=rf'$\sigma={sigma:.2f}$',
                    alpha=opacity
                )
                if alpha == -1:
                    axin.semilogx(
                        lambda_arr[1:18], N[1:18], color=colors[i], marker=markers[i], 
                        linestyle=linestyles[j], linewidth=0.85, mfc='white', markersize=2.75,
                        alpha=opacity
                    )
                    
                # axes[2].semilogx(
                #     lambda_arr, R, color=colors[i], marker=markers[i], linestyle=linestyles[j],
                #     linewidth=0.85, mfc='white', markersize=4, label=rf'$\sigma={sigma:.2f}$'
                # )
                # Determine λ* for which the system is fragile
                if alpha == -1:
                    __M = args.rho - _M                     
                    _max_idx = np.argmax(__M>0.98*args.rho, axis=0)
                    lambda_u = 10**(np.mean(np.log10(lambda_arr[_max_idx])))
                elif alpha <= 1.1:
                    __M = _M[::-1,:]
                    _min_idx = len(lambda_arr) - np.argmax(__M>0.98*args.rho, axis=0) - 1
                    lambda_o = 10**(np.mean(np.log10(lambda_arr[_min_idx])))
                else:
                    pass 
                # Determine λ for which the system is healthy
            lambda_star[i] = 10**((np.log10(lambda_u)+np.log10(lambda_o))/2)
        # Compute average λ* for all σ
        mean_lambda_star = 10**(np.mean(np.log10(lambda_star)))
        # Indicate it in axis
        axes[1].text(
            0.8*mean_lambda_star, 0.04, r'$\hat\lambda\approx{:.2f}$'.format(mean_lambda_star), 
            fontsize=12, ha='center', rotation=90
        )
        
        # Limits, labels, etc
        # Inset axes
        _, connectors = axes[0].indicate_inset_zoom(axin, ec='k', alpha=0.5)
        axin.tick_params(axis='both', labelsize=8)
        axin.set_xlim(min(lambda_arr), 0.15)
        axin.set_ylim(0,0.1)
        axin.yaxis.set_minor_locator(MultipleLocator(0.025))
        axin.text(
            0.05, 0.95, r'$\alpha\rightarrow \infty$', ha='left', va='top',
            fontsize=10, transform=axin.transAxes
        )
        # Legend        
        sigma_lines = [
            Line2D(
                [],[], color=colors[i], linestyle='none', marker=markers[i], 
                mfc='white', markersize=3.5
            ) for i in range(len(sigma_arr))
        ]
        sigma_labels = [rf'$\sigma\!=\!{x:.2f}$' for x in sigma_arr]
        alpha_lines = [
            Line2D(
                [],[], color='k', linestyle=linestyles[i], linewidth=0.85
            ) for i in range(len(alpha_arr))
        ]
        alpha_labels = [r'$\alpha\!\rightarrow\!1$', r'$\alpha\!\rightarrow\! \infty$']
        # Axes
        ylims = [1.75*args.rho, 1.1*args.rho]
        ylabels = [r'$N$', r'$M$']
        for i, ax in enumerate(axes):
            # Plot helper line at λ=λ*
            ax.semilogx(
                [mean_lambda_star, mean_lambda_star], [0,ylims[i]], 
                color='k', linestyle=':', linewidth=0.85, dashes=(1,1)
            )
            # Limits
            ax.set_xlim(min(lambda_arr), max(lambda_arr))
            ax.set_ylim(0, ylims[i])
            ax.set_ylabel(ylabels[i], fontsize=14)
            ax.text(
                0.0, 1.01, figbflabels[i], ha='left', va='bottom', 
                fontsize=14, transform=ax.transAxes
            )
            yloc = 0.05 if i == 0 else 0.025
            ax.yaxis.set_minor_locator(MultipleLocator(yloc))
            # Legend
            if i == 1:
                ax.set_xlabel(r'$\hat{\lambda}$', fontsize=14)
                sigma_legend = plt.legend(
                    sigma_lines, sigma_labels, 
                    loc='lower left', fontsize=9.5, frameon=False, borderaxespad=-0.15,
                    labelspacing=0, handlelength=0.5, handletextpad=0.2
                )
                alpha_legend = plt.legend(
                    alpha_lines, alpha_labels, 
                    loc='upper right', fontsize=10, frameon=False, borderaxespad=-0.15,
                    labelspacing=0.1, handlelength=0.85, handletextpad=0.2
                )
                plt.gca().add_artist(sigma_legend)
                plt.gca().add_artist(alpha_legend)
        
        # Save
        self.figdict[f'population_densities_sigma_H{args.H:.4f}_rho{args.rho:.2f}'] = fig 
    
    def plot_preferred_fragmentation(self, args):
        """ Plot preferred fragmentation """
        L = 2**args.m
        # Specify directory        
        _dir = args.rdir+'sllvm/ideal_fragmentation/'
        _rdir = args.rdir+'sllvm/ideal_fragmentation/{L:d}x{L:d}/'.format(L=L)
        # Load variable arrays
        alpha_arr = np.loadtxt(_dir+'alpha.txt')
        H_arr = np.loadtxt(_dir+'H.txt')
        # Initialize figure
        fig, axes = plt.subplots(2,1, figsize=(2.5,4.5/4*3), sharex=True)
        # Load data
        suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}_mu{:.4f}_' \
            'Lambda{:.4f}_lambda{:.4f}_sigma{:.4f}'.format(
            args.T, args.N0, args.M0,
            args.rho, args.mu, args.Lambda_, args.lambda_, args.sigma
        )
        _alphastar_N = np.load(_rdir+f'alphastar_N{suffix}.npy')
        _alphastar_R = np.load(_rdir+f'alphastar_R{suffix}.npy')
        alphastar_N = np.mean(_alphastar_N, axis=1)
        alphastar_R = np.mean(_alphastar_R, axis=1)
        alphastar_Nstd = np.std(_alphastar_N, axis=1)
        alphastar_Rstd = np.std(_alphastar_R, axis=1)
        _Hstar_N = np.load(_rdir+f'Hstar_N{suffix}.npy')
        _Hstar_M = np.load(_rdir+f'Hstar_M{suffix}.npy')
        _Hstar_R = np.load(_rdir+f'Hstar_R{suffix}.npy')
        Hstar_N = np.nanmean(_Hstar_N, axis=1)
        Hstar_M = np.nanmean(_Hstar_M, axis=1)
        Hstar_R = np.nanmean(_Hstar_R, axis=1)
        Hstar_Nstd = np.std(_Hstar_N, axis=1)
        Hstar_Mstd = np.std(_Hstar_M, axis=1)
        Hstar_Rstd = np.std(_Hstar_R, axis=1)
        # Print α* for some H
        H = [0.01, 0.2, 0.5, 1]
        Hidxs = [np.flatnonzero(H_arr==h) for h in H]
        for i, idx in enumerate(Hidxs):
            print(H[i], alphastar_R[idx])
        # Print H* for some α
        alpha = [1.1, 2.0, 2.5, 3.0]
        alphaidxs = [np.flatnonzero(alpha_arr==a) for a in alpha]
        for i, idx in enumerate(alphaidxs):
            print(alpha[i], Hstar_N[idx])
        # Plot data
        # axes[0].errorbar(
        #     H_arr, alphastar_N,# yerr=alphastar_Nstd, capsize=1.5,
        #     color='k', marker='o', mfc='white', markersize=3.5,
        #     linewidth=0.85, label=r'$\alpha^*_N$'
        # )
        # axes[0].errorbar(
        #     H_arr, alphastar_R,# yerr=alphastar_Rstd, capsize=1.5,
        #     color='navy', marker='D', mfc='white', markersize=3.5,
        #     linestyle='--', linewidth=0.85, label=r'$\alpha^*_\mathcal{R}$'
        # )
        # Store value of H and optimal responses
        if args.compute:
            np.savetxt(_dir+'H_arr.txt.txt', H_arr, fmt='%.4f')
            np.savetxt(_dir+'alphastar_N.txt', alphastar_N, fmt='%.4f')
            np.savetxt(_dir+'alphastar_R.txt', alphastar_R, fmt='%.4f')
        axes[0].errorbar(
            alpha_arr, Hstar_N,# yerr=Hstar_Nstd, capsize=1.5,
            color='k', marker='s', mfc='white', markersize=3.5,
            linestyle='-', linewidth=0.85, label=r'$H^*_N$'
        )
        axes[0].errorbar(
            alpha_arr, Hstar_M,# yerr=Hstar_Mstd, capsize=1.5,
            color='navy', marker='D', mfc='white', markersize=3.5,
            linestyle=':', linewidth=0.85, label=r'$H^*_M$'
        )
        axes[0].errorbar(
            alpha_arr, Hstar_R,# yerr=Hstar_Rstd, capsize=1.5,
            color='firebrick', marker='o', mfc='white', markersize=3.5,
            linestyle='--', linewidth=0.85, label=r'$H^*_\mathcal{R}$'
        )
        # Store values of alpha_arr and the (optimal) fragmentations 
        if args.compute:
            np.savetxt(_dir+'alpha_arr.txt', alpha_arr, fmt='%.4f')
            np.savetxt(_dir+'Hstar_N.txt', Hstar_N, fmt='%.4f')
            np.savetxt(_dir+'Hstar_M.txt', Hstar_M, fmt='%.4f')
            np.savetxt(_dir+'Hstar_R.txt', Hstar_R, fmt='%.4f')
        plt.show()
        exit()
        # H_plot = [0.01, 0.2, 0.5, 1.]
        # for __H in H_plot:
        #     print(__H, alphastar_R[np.argwhere(H_arr==__H)])
        # Plot helpers
        # _latticedir = args.ddir+'landscapes/'
        # suffix = '_{L:d}x{L:d}_H{H:.3f}_rho{rho:.3f}'.format(L=L, H=0.99, rho=args.rho)
        # lattice_lo = np.load(_latticedir+"lattice{suffix:s}.npy".format(suffix=suffix))
        # suffix = '_{L:d}x{L:d}_H{H:.3f}_rho{rho:.3f}'.format(L=L, H=0.01, rho=args.rho)
        # lattice_hi = np.load(_latticedir+"lattice{suffix:s}.npy".format(suffix=suffix))
        # axin_hi = axes[0].inset_axes([-0.0325,0.68,0.3,0.3])
        # axin_hi.imshow(lattice_hi, cmap='Greys')
        # axin_lo = axes[0].inset_axes([0.735,0.68,0.3,0.3])
        # axin_lo.imshow(lattice_lo, cmap='Greys')
        # axin_hi.text(
        #     1.2,0.5, r'$H\rightarrow 0$', fontsize=9, rotation=90, 
        #     ha='center', va='center', transform=axin_hi.transAxes
        # )
        # axin_lo.text(
        #     -0.2,0.5, r'$H\rightarrow 1$', fontsize=9, rotation=90, 
        #     ha='center', va='center', transform=axin_lo.transAxes
        # )
        # for axin in [axin_hi, axin_lo]:
        #     axin.xaxis.set_visible(False)
        #     axin.yaxis.set_visible(False)

        # Plot population dynamics in for ideal fragmentation
        suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}' \
            '_Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
            args.T, args.N0, args.M0, args.rho, 
            args.Lambda_, args.lambda_, args.mu, args.sigma
        )
        _N = np.load(_rdir+f'N_optimal_fragmentation_N{suffix}.npy')
        N = np.mean(_N, axis=1) / L**2 
        axes[1].plot(
            alpha_arr[1:], N[1:], color='k', marker='s', mfc='white', markersize=3,
            linewidth=0.85, markevery=2, label=r'$N^*_{N}$'
        )
        _M = np.load(_rdir+f'M_optimal_fragmentation_N{suffix}.npy')
        M = np.mean(_M, axis=1) / L**2
        axes[1].plot(
            alpha_arr[1:], M[1:], color='k', marker='s', mfc='lightgrey', markersize=3,
            linestyle='-', linewidth=0.85, markevery=2, label=r'$M^*_{N}$'
        )
        _N = np.load(_rdir+f'N_optimal_fragmentation_M{suffix}.npy')
        N = np.mean(_N, axis=1) / L**2 
        axes[1].plot(
            alpha_arr[1:], N[1:], color='navy', marker='D', mfc='white', markersize=3,
            linestyle=':', linewidth=0.85, markevery=2, label=r'$N^*_{M}$'
        )
        _M = np.load(_rdir+f'M_optimal_fragmentation_M{suffix}.npy')
        M = np.mean(_M, axis=1) / L**2
        axes[1].plot(
            alpha_arr[1:], M[1:], color='navy', marker='D', mfc='steelblue', markersize=3,
            linestyle=':', linewidth=0.85, markevery=2, label=r'$M^*_{M}$'
        )
        # _N = np.load(_rdir+f'N_optimal_fragmentation_R{suffix}.npy')
        # N = np.mean(_N, axis=1) / L**2 
        # axes[1].plot(
        #     alpha_arr[1:], N[1:], color='firebrick', marker='o', mfc='navy', markersize=3,
        #     linestyle=':', linewidth=0.85, markevery=2
        # )
        # _M = np.load(_rdir+f'M_optimal_fragmentation_R{suffix}.npy')
        # M = np.mean(_M, axis=1) / L**2
        # axes[1].plot(
        #     alpha_arr[1:], M[1:], color='firebrick', marker='o', mfc='white', markersize=3,
        #     linestyle=':', linewidth=0.85, markevery=2
        # )

        # Limits, labels, etc
        axes[0].set_ylabel(r'$H^*$', fontsize=14)
        axes[1].set_ylabel(r'population density', fontsize=11)
        axes[1].set_xlabel(r'$\alpha$', fontsize=14)
        ylim = [1, 1.05*args.rho]
        yloc = [0.1, 0.025]
        for i, ax in enumerate(axes):
            if i == 0:
                ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
            else:
                ax.set_yticks([0,0.1,0.2])
            ax.set_xlim([1,3])
            ax.set_ylim([0,ylim[i]])
            ax.set_ylabel(r'$H^*$', fontsize=14)
            ax.xaxis.set_minor_locator(MultipleLocator(0.125))
            ax.yaxis.set_minor_locator(MultipleLocator(yloc[i]))
            ncol = 1 if i == 0 else 2
            loc = 'lower right' if i==0 else 'upper right'

            handles, labels = ax.get_legend_handles_labels()
            if i == 1:
                order = [0,2,1,3]
                handles, labels = [handles[k] for k in order], [labels[k] for k in order]
            legend = ax.legend(
                handles, labels, loc=loc, fontsize=11, handlelength=1.5, handletextpad=0.4,
                borderaxespad=0., labelspacing=0.5, frameon=False,
                ncol=ncol, columnspacing=0.2
            )
            plt.setp(legend.get_texts(), ha='left', va='center')
            ax.text(
                0.01, 1.01, figbflabels[i], ha='left', va='bottom',
                fontsize=14, transform=ax.transAxes
            )
        # Save
        self.figdict[f'optimalvalues_rho{args.rho}'] = fig

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
    # Flight length related plots #
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
    
    #######################################
    # Environmental metrics related plots #
    def plot_effective_habitat(self, args):
        """ Plot effective habitat as a function of the Levy parameter """
        L = 2**args.m
        # Specify directory        
        _dir = args.rdir+'sllvm/habitat_loss/'
        _rdir = args.rdir+'sllvm/habitat_loss/{L:d}x{L:d}/'.format(L=L)
        if args.compute:
            _rdir = _rdir+'fit/'
            _dir = _rdir
        
        # Load variables
        alpha_arr = np.loadtxt(_dir+'alpha.txt')
        H_arr = [0.01, 0.2, 0.5, 1]
        fitax = np.linspace(1, max(alpha_arr), 250)
        # Initialize figure
        # fig, axes = plt.subplots(2,1, figsize=(2.75,5.25/4*3))
        fig, axes = plt.subplots(1, 2, figsize=(2*2.75,2.25/4*3), sharey=True)
        fig.subplots_adjust(hspace=0.5)
        # Load data & plot as a function of alpha
        for i, H in enumerate(H_arr):
            label = r'$H\rightarrow 1$' if H==1 else rf'$H={H:.2f}$'
            # Load data
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, H, 
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma
            )
            # Load
            _nI = np.load(_rdir+f'num_isolated_patches{suffix}.npy')
            rho_eff = 1 - np.mean(_nI, axis=1) / (args.rho*L**2)
            # Plot
            popt, _ = curve_fit(Plotter.sigmoid, alpha_arr, rho_eff)
            fit = np.minimum(1, Plotter.sigmoid(fitax, *popt))
            axes[0].plot(fitax, fit, color=colors[i], linewidth=0.85)
            axes[0].plot(
                [],[], color=colors[i], marker=markers[i], mfc='white',
                markersize=3, linewidth=0.85, label=label
            )
            axes[0].plot(
                alpha_arr, rho_eff, color=colors[i], marker=markers[i], mfc='white',
                markersize=3, linestyle='none'
            )
        # Load data & plot as a function of H
        alpha_arr = [1.1, 1.5, 2.0, 2.5, 3.0]
        H_arr = np.loadtxt(_dir+'H.txt')
        fitax = np.linspace(0, max(H_arr), 250)
        for i, alpha in enumerate(alpha_arr):
            label = r'$\alpha={:.1f}$'.format(alpha)
            # Load data
            suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
                args.T, args.N0, args.M0, args.rho, 
                args.Lambda_, args.lambda_, args.mu, args.sigma, alpha
            )
            # Load
            _nI = np.load(_rdir+f'num_isolated_patches{suffix}.npy')
            rho_eff = 1 - np.mean(_nI, axis=1) / (args.rho*L**2)
            # Plot
            popt, _ = curve_fit(Plotter.sigmoid, H_arr, rho_eff)
            fit = np.minimum(1, Plotter.sigmoid(fitax, *popt))
            axes[1].plot(fitax, fit, color=colors[i], linewidth=0.85)
            axes[1].plot(
                [],[], color=colors[i], marker=markers[i], mfc=lightcolors[i],
                markersize=3.5, linewidth=0.85, label=label
            )
            axes[1].plot(
                H_arr, rho_eff, color=colors[i], marker=markers[i], mfc=lightcolors[i],
                markersize=3.5, linestyle='none'
            )
        # Helper lines
        # ax.plot([1,1], [0.4,1.05], color='k', linewidth=0.85)
        # ax.plot([1,1], [0.,0.035], color='k', linewidth=0.85)
        xlims = [[1,max(alpha_arr)], [0, 1]]
        xlabels = [r'$\alpha$', r'$H$']
        # Limits, labels, etc
        for i, ax in enumerate(axes):
            ax.set_xlim(xlims[i])
            xmloc = 0.25 if i == 0 else 0.1
            ax.xaxis.set_minor_locator(MultipleLocator(xmloc))
            # ax.set_xticks([1+0.5*i for i in range(7)])
            ax.set_ylim(0, 1.01)
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.set_yticks([0+0.2*i for i in range(6)])
            ax.set_xlabel(xlabels[i], fontsize=14)
            if i == 0:
                ax.set_ylabel(r'$\rho_{\text{eff}}$', fontsize=14)
            ncol = 1 if i == 0 else 2
            loc = 'lower left' if i == 0 else 'lower right'
            (handles, labels) = ax.get_legend_handles_labels()
            if i == 1:                
                handles.insert(0, plt.Line2D([],[], linestyle='none'))
                labels.insert(0, '')
            legend = ax.legend(
                handles, labels, loc=loc, fontsize=9.5, labelspacing=0.1, handlelength=1, 
                borderaxespad=-0.15, handletextpad=0.1, frameon=False,
                ncol=ncol, columnspacing=0.15
            )
            # plt.setp(legend.get_texts(), ha='center', va='center')
            ax.text(
                0., 1.01, figbflabels[i], ha='left', va='bottom', 
                fontsize=14, transform=ax.transAxes
            )
        fig.subplots_adjust(wspace=0.15, hspace=0.5)
        # Save
        self.figdict[f'rho_effective_rho{args.rho:.2f}'] = fig
    
    def plot_heatmap(self, args):
        """ Plot a heatmap of population densities and species richness for 
            different α and H, as to gain insight into complex interactions 
        """
        L = 2**args.m 
        # Specify directory
        _dir = args.rdir+'sllvm/heatmap/'
        _rdir = args.rdir+'sllvm/heatmap/{L:d}x{L:d}/'.format(L=L)
        # Load or specify variable arrays
        alpha_arr = np.loadtxt(_dir+'alpha.txt')
        H_arr = np.loadtxt(_dir+'H.txt')
        # Initialize figure
        fig, axes = plt.subplots(1, 3, figsize=(9,3), tight_layout=True)
        # Get colormap
        cmap = plt.cm.get_cmap('YlGn')
        # cmap = plt.cm.get_cmap('RdYlGn')
        # _colors = ['gold', 'navy', 'lightgreen']
        # cvals = np.linspace(0, 1, len(_colors))
        # cvals = [0, 0.05, 0.1, 0.2, 0.5, 1.]
        # colors = ['k', 'darkmagenta', 'yellow', 'green', 'navy', 'aqua']
        # cvals = [0, 0.1, 0.2, 0.4, 0.7, 1]
        # colors = ['red', 'yellow', 'green', 'lightseagreen', 'navy', 'lightskyblue']
        # cvals[0] = 0
        # norm = plt.Normalize(min(cvals), max(cvals))
        # tuples = list(zip(map(norm,cvals), _colors))
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', tuples)
        
        # Load & plot 
        suffix = (
                '_T{:d}_N{:d}_M{:d}_rho{:.3f}_mu{:.4f}'
                '_Lambda{:.4f}_lambda{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, args.rho, args.mu,
                args.Lambda_, args.lambda_, args.sigma
            )
        )
        _N = np.load(_rdir+f'N_heatmap{suffix}.npy')
        _M = np.load(_rdir+f'M_heatmap{suffix}.npy')
        _R = np.load(_rdir+f'R_heatmap{suffix}.npy')
        N = np.nanmean(_N, axis=2) / L**2 
        M = np.nanmean(_M, axis=2) / L**2 
        R = np.nanmean(_R, axis=2) / L**2 
        N /= np.max(N) 
        M /= np.max(M) 
        R /= np.max(R)
        # Plot
        imN = axes[0].imshow(
            N, aspect='auto', cmap=cmap, origin='lower', vmin=0, vmax=1,
            extent=(min(alpha_arr), max(alpha_arr), min(H_arr), max(H_arr)),
            interpolation='spline16', alpha=0.75
        )
        imM = axes[1].imshow(
            M, aspect='auto', cmap=cmap, origin='lower', vmin=0, vmax=1,
            extent=(min(alpha_arr), max(alpha_arr), min(H_arr), max(H_arr)),
            interpolation='spline16', alpha=0.75
        )
        imR = axes[2].imshow(
            R, aspect='auto', cmap=cmap, origin='lower', vmin=0, vmax=1,
            extent=(min(alpha_arr), max(alpha_arr), min(H_arr), max(H_arr)),
            interpolation='spline16', alpha=0.75
        )
        # Get maximum
        for i, Z in enumerate([N, M, R]):
            idx = np.where(Z==np.max(Z))
            _H = H_arr[idx[0]]
            _alpha = alpha_arr[idx[1]]
            axes[i].plot(_alpha, _H, marker='o', linestyle='none', color='red', markersize=2.5)
            axes[i].annotate(
                r'', (_alpha+0.01,_H-0.01), xytext=(_alpha+0.15, _H-0.1), 
                arrowprops=dict(arrowstyle='->', color='r')
            )
        # Limits, labels, etc
        ims = [imN, imM, imR]
        cmax = [0.2, 0.2, 0.2]
        labels = [r'$N/N_{max}$', r'${M}$', r'${\mathcal{R}}$']
        for i, ax in enumerate(axes):
            ax.set_xlabel(r'$\alpha$', fontsize=16)
            if i == 0:
                ax.set_ylabel(r'$H$', fontsize=16)
            ax.text(
                0.5, 1.035, labels[i], fontsize=16, ha='center', transform=ax.transAxes
            )
            ax.set_xticks([1.0,1.5,2.0,2.5,3.0])
            ax.set_yticks([0.01,0.2,0.4,0.6,0.8,1.0])
            ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0])
            # Colorbar 
            if i == len(axes)-1:
                cbar = fig.colorbar(ims[i], ax=ax, fraction=0.04, pad=0.04)
            # cbar.set_clim(0, cmax[i])
            # cbar.set_ticks(np.linspace(0,cmax[i],5))
            # cbar.set_ticklabels([r'%.2f'%(k) for k in np.linspace(0,cmax[i],5)])
        # Save
        self.figdict['alpha_H_heatmap'] = fig 

    def plot_habitat_loss_probability(self, args):
        """ Plot the habitat loss as the probability of a patch to be (indefinetly) 
            depleted as a function of patch size, for different Levy parameters
        """         
        L = 2**args.m
        # Specify directory        
        _dir = args.rdir+'sllvm/habitat_loss/'
        _rdir = args.rdir+'sllvm/habitat_loss/{L:d}x{L:d}/'.format(L=L)
        _sizedir = args.ddir+'patch_distribution/{L:d}x{L:d}/'.format(L=L)
        # Specify variables
        alpha_arr = [1.1, 1.5, 2.0, 2.5]
        patchbins = np.logspace(0, np.log10(args.rho*L**2+1), num=args.nbins, dtype=np.int64)
        patchbins = np.unique(patchbins)
        patchbins = patchbins / (args.rho*L**2)
        fitax = np.logspace(0, np.log10(args.rho*L**2+1), num=10*args.nbins) / (args.rho*L**2)
        # Initialize figure
        # fig, axes = plt.subplots(2,1, figsize=(4.25/4*3,4.5))
        # fig, ax = plt.subplots(1,1, figsize=(2.75,2.25/4*3))
        fig, axes = plt.subplots(1, 2, figsize=(2*2.75,2.25/4*3), sharey=True)
        ax = axes[0]
        axsi = axes[1]
        # figsi, axsi = plt.subplots(1,1, figsize=(2.75,2.25/4*3))
        # Load patch size xmax
        _max_sizes = np.load(_sizedir+'patch_size_H{:.3f}_rho{:.3f}.npy'.format(args.H, args.rho))
        _max_size = np.mean(_max_sizes) / (args.rho*L**2)
        fitax = fitax[:np.argmax(fitax>_max_size)]
        # Load & plot for different α
        __i = 0
        for i, alpha in enumerate(alpha_arr):
            # Load 
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
                args.T, args.N0, args.M0, args.H,
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma, alpha
            )
            _P = np.load(_rdir+f'prob_patch_depletion{suffix}.npy')
            P = np.mean(_P, axis=1)
            # Fix color as to accomodate P*
            if colors[i] == 'firebrick':
                __i += 1
            # color = colors[i] if colors[i] != 'firebrick' else 'darkgreen'
            # lightcolor = lightcolors[i] if colors[i] != 'firebrick' else 'mediumseagreen'
            # marker = markers[i+1] if colors[i] != 'firebrick' else markers[i]
            # Compute fit 
            popt, _ = curve_fit(Plotter.tanh, np.log10(patchbins), P)            
            fit = np.minimum(1, Plotter.tanh(np.log10(fitax), *popt))
            # Plot
            ax.plot(
                [],[], color=colors[__i], marker=markers[__i], mec=colors[__i], mfc=lightcolors[__i],
                markersize=3.5, linestyle='--', linewidth=0.85, label=rf'$\alpha={alpha:.1f}$'
            )
            ax.semilogx(fitax, fit, color=colors[__i], linewidth=0.85, linestyle='--')
            ax.semilogx(
                patchbins, P, color=colors[__i], marker=markers[__i], mfc=lightcolors[__i],
                mec=colors[__i], markersize=3.5, linestyle='none'
            )
            __i += 1
        # Plot P for optimal α
        suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_' \
            'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
            args.T, args.N0, args.M0, args.H,
            args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma, 1.195
        ) 
        _Pstar = np.load(_rdir+f'prob_patch_depletion{suffix}.npy')
        Pstar = np.mean(_Pstar, axis=1)
        # Compute fit 
        popt, _ = curve_fit(Plotter.tanh, np.log10(patchbins), Pstar)
        fit = np.minimum(1, Plotter.tanh(np.log10(fitax), *popt))
        ax.plot(
            [],[], color='firebrick', marker='D', mfc='white',
            markersize=3.5, linestyle='-', linewidth=0.85
        )
        ax.semilogx(fitax, fit, color='firebrick', linewidth=0.85, linestyle='-')
        ax.semilogx(
            patchbins, Pstar, color='firebrick', marker='D', mfc='white',
            markersize=3.5, linestyle='none'
        )
        ax.text(
            4.5e-4, 0.35, rf'$\alpha={1.195:.2f}$', fontsize=10, rotation=-73,
            ha='center', va='center', color='firebrick'
        )

        # Add annotated line at x=xmax
        ax.semilogx(
            [_max_size,_max_size], [0, 0.15], color='k', linewidth=0.7, 
            linestyle='--'
        )
        ax.text(
            _max_size, 0.18, r'$x_{\text{max}}$', rotation=90, va='bottom', ha='center',
            fontsize=10
        )
        # Load & plot for different H 
        # H_arr = np.loadtxt(_dir+'H.txt')
        H_arr = [0.01, 0.2, 0.5, 1]
        alpha_arr = [1.472, 1.316, 1.195, 1.117]
        _min_fitax = np.inf
        for i, (H, alpha) in enumerate(zip(H_arr,alpha_arr)):
            # Load patch size
            _max_sizes = np.load(_sizedir+'patch_size_H{:.3f}_rho{:.3f}.npy'.format(H, args.rho))
            _max_size = np.mean(_max_sizes) / (args.rho*L**2)
            # Normalize bins by maximum encountered patch size
            _patchbins = patchbins / _max_size
            fitax = np.logspace(np.log10(1/_max_size), np.log10(args.rho*L**2+1), num=10*args.nbins) / (args.rho*L**2)
            _min_fitax = min(_min_fitax, np.min(fitax))
            # Load
            suffix = '_T{:d}_N{:d}_M{:d}_H{:.4f}_rho{:.3f}_' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}_alpha{:.3f}'.format(
                args.T, args.N0, args.M0, H,
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma, alpha
            )
            _P = np.load(_rdir+f'prob_patch_depletion{suffix}.npy')
            P = np.mean(_P, axis=1)
            # Compute fit 
            popt, _ = curve_fit(Plotter.tanh, np.log10(_patchbins), P)
            fit = np.minimum(1, Plotter.tanh(np.log10(fitax), *popt))
            # Omit any other bins above that, which can be safely done as P=0 for any patch
            # larger than the (studied) patch size anyways
            _patchbins = _patchbins[:np.argmax(_patchbins>1)]
            P = P[:len(_patchbins)]
            # Define label 
            # _label = rf'H&={H:.2f}' + r'\\' + rf'\alpha&={alpha:.2f}'
            # label = r'\begin{align*}' + _label + r'\end{align*}'
            label = rf'$H={H:.2f}$' if H < 1 else r'$H\rightarrow 1$'
            # Plot
            axsi.plot(
                [], [], color=colors[i], marker=markers[i], mfc='white', 
                mec=colors[i], markersize=3.5, label=label, linewidth=0.85
            )      
            axsi.plot(fitax, fit, color=colors[i], linewidth=0.85)
            axsi.semilogx(
                _patchbins, P,  color=colors[i], marker=markers[i], mfc='white', 
                markersize=3.5, linestyle='none'
            )
        # Helper lines
        alpha_pos = [2.2e-4, 0.8e-3, 3.3e-3, 2.1e-2][::-1]
        for i, pos in enumerate(alpha_pos):
            axsi.text(
                pos, 0.35, rf'$\alpha={alpha_arr[i]:.2f}$', fontsize=10, rotation=-73,
                ha='center', va='center', color=colors[i]
            )
        for i, ax in enumerate([ax,axsi]):
            if i == 0:
                ax.set_ylabel(r'$P_d$', fontsize=14)
            # Limits, labels, etc
            xmax = 1
            ax.set_xlim(1e-5, xmax)
            ax.set_ylim(0, 1.02)
            xlab = r'$x / \rho L^2$' if i == 0 else r'$x/x_{\text{max}}$'
            ax.set_xlabel(xlab, fontsize=14)
            ax.legend(
                loc='upper right', fontsize=9.5, labelspacing=0.1, handlelength=1.4,
                borderaxespad=-0.1, handletextpad=0.2, frameon=False
            )
            # ax.text(
            #     0., 1.01, figbflabels[i], ha='left', va='bottom', 
            #     fontsize=14, transform=ax.transAxes
            # )
            # Add annotated line at x=1
            if i == 0:
                xmin = min(patchbins) if i == 0 else _min_fitax
                ax.semilogx(
                    [xmin,xmin], [0,1.05], color='k', linestyle=':', linewidth=0.65
                )
                xminloc = 0.1 if i == 0 else 0.05
                ax.text(
                    xmin, 0.1, r'$x=1$', rotation=90, ha='center', va='bottom',
                    fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='none', pad=0.1)
                )
            ax.text(
                0., 1.01, figbflabels[i], ha='left', va='bottom', 
                fontsize=14, transform=ax.transAxes
            )
            # if i == 0:            
            #     ax.semilogx(
            #         [max(patchbins),max(patchbins)], [0,0.5], color='k', linestyle=':', 
            #         linewidth=0.65
            #     )
            # Log formatter
            ax.set_xticks([10**i for i in range(-5,1,1)])
            locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1, numticks=100)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        
        # fig.subplots_adjust(wspace=0.15, hspace=0.5)
        # Save 
        self.figdict[f'habitat_loss_rho{args.rho:.2f}'] = fig 
        # self.figdict[f'habitat_loss_optimal_response_rho{args.rho:.2f}'] = figsi

    def plot_effective_habitat_optimal_response(self, args):
        """ Plot the effective habitat under optimal predator or landscape responses """
        L = 2**args.m
        # Specify directory        
        _dir = args.rdir+'sllvm/habitat_loss/'
        _rdir = args.rdir+'sllvm/habitat_loss/{L:d}x{L:d}/'.format(L=L)
        if args.compute:
            _rdir = _rdir+'fit/'
            _dir = _rdir
        
        # Load variables
        # alpha_arr = np.loadtxt(_dir+'alpha.txt')
        H_arr = np.loadtxt(_dir+'H.txt')
        fitax = np.linspace(0.01,1, 250)
        # Initialize figure
        # fig, axes = plt.subplots(2,1, figsize=(2.5,4.5/4*3), sharex=True)
        fig, axes = plt.subplots(1, 2, figsize=(2*2.75,2.25/4*3))
        fig.subplots_adjust(wspace=0.35)
        # Load data & plot
        # Load data
        suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}_' \
            'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
            args.T, args.N0, args.M0, 
            args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma
        )
        # Define fitting functions
        population_curve = lambda x, a, b, c: a*x**b + c

        # Load effective habitat and corresponding densities
        vals = ['R', 'N']
        subscripts = ['\mathcal{R}', 'N']
        for i, val in enumerate(vals):
            _rho_eff = np.load(_rdir+f'effective_density_optimal_response_{val}{suffix}.npy')
            rho_eff = np.mean(_rho_eff, axis=1)
            _N = np.load(_rdir+f'N_optimal_response_{val}{suffix}.npy')
            N = np.mean(_N, axis=1) / L**2
            _M = np.load(_rdir+f'M_optimal_response_{val}{suffix}.npy')
            M = np.mean(_M, axis=1) / L**2
            # Load & plot effective habitat density
            popt, _ = curve_fit(Plotter.sigmoid, H_arr, rho_eff)
            fit = np.minimum(1, Plotter.sigmoid(fitax, *popt))
            axes[0].plot(fitax, fit, color=colors[2*i], linestyle=linestyles[i], linewidth=0.85)
            axes[0].plot(
                [],[], color=colors[2*i], marker=markers[i], mfc='white', linestyle=linestyles[i],
                markersize=3.5, linewidth=0.85, label=rf'$\alpha=\alpha^*_{subscripts[i]}$'
            )
            axes[0].plot(
                H_arr, rho_eff, color=colors[2*i], marker=markers[i], mfc='white',
                markersize=3.5, linestyle='none'
            )
            # Load & plot populations under optimal responses
            axes[1].plot(
                [],[], color=colors[2*i], marker=markers[i], mfc='white', linestyle='-',
                markersize=3.5, linewidth=0.85, label=rf'$N^*_{subscripts[i]}$'
            )
            popt, _ = curve_fit(population_curve, H_arr, N)
            fit = np.minimum(1, population_curve(fitax, *popt))
            axes[1].plot(fitax, fit, color=colors[2*i], linestyle='-', linewidth=0.85)
            axes[1].plot(
                H_arr, N, color=colors[2*i], marker=markers[i], mfc='white', 
                markersize=3, linestyle='none'
            )
            axes[1].plot(
                [],[], color=colors[2*i], marker=markers[i+2], mfc='white', linestyle='--',
                mec=colors[2*i], markersize=3.5, linewidth=0.85, label=rf'$M^*_{subscripts[i]}$'
            )
            try:
                popt, _ = curve_fit(Plotter.sigmoid, H_arr, M)
                fit = np.minimum(1, Plotter.sigmoid(fitax, *popt))
                axes[1].plot(fitax, fit, color=colors[2*i], linestyle='--', linewidth=0.85, dashes=(2,1))
            except:
                pass 
            axes[1].plot(
                H_arr, M, color=colors[2*i], marker=markers[i+2], mfc='white', mec=colors[2*i],
                markersize=3, linestyle='none'
            )
            
        # Limits, labels, etc
        axes[0].set_ylabel(r'$\rho_{\text{eff}}^*$', fontsize=14)
        axes[1].set_ylabel(r'population density', fontsize=11)
        axes[1].set_xlabel(r'$H$', fontsize=14)
        axes[0].set_ylim(0.5,1.015)
        axes[0].set_yticks([0.5,0.6,0.7,0.8,0.9,1.0])
        axes[1].set_ylim(0,0.15)
        axes[1].set_xticks([0,0.2,0.4,0.6,0.8,1.])
        yloc = [0.05, 0.025]
        for i, ax in enumerate(axes):
            ax.set_xlim(0,1)
            ncol = 1 if i == 0 else 2
            handles, labels = ax.get_legend_handles_labels()
            loc = 'lower right' if i == 1 else 'upper right'
            bbox = (0,0,1,1) if i == 1 else (0,0,1,0.925)
            if i == 1:
                order = [0,2,1,3]
                handles, labels = [handles[k] for k in order], [labels[k] for k in order]
            ax.legend(
                handles, labels, loc=loc, handlelength=1.5, handletextpad=0.2, frameon=False,
                borderaxespad=0.1, labelspacing=0.2, fontsize=11, ncol=ncol, 
                columnspacing=0.3, bbox_to_anchor=bbox
            )
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(axis='both', labelsize=9)
            ax.text(
                0.0, 1.01, figbflabels[i], ha='left', va='bottom',
                fontsize=14, transform=ax.transAxes
            )
            ax.yaxis.set_minor_locator(MultipleLocator(yloc[i]))

        # Load data on optimal response
        # Specify directory        
        _dir = args.rdir+'sllvm/optvalues/'
        _rdir = args.rdir+'sllvm/optvalues/{L:d}x{L:d}/'.format(L=L)
        # Load variable arrays
        H_arr = np.loadtxt(_dir+'H.txt')
        # Initialize axis
        axin = axes[0].inset_axes([0.4, 0.15, 0.55, 0.4])
        # Load data
        suffix = '_T{:d}_N{:d}_M{:d}_rho{:.3f}_mu{:.4f}_' \
            'Lambda{:.4f}_lambda{:.4f}_sigma{:.4f}'.format(
            args.T, args.N0, args.M0,
            args.rho, args.mu, args.Lambda_, args.lambda_, args.sigma
        )
        _alphastar_N = np.load(_rdir+f'alphastar_N{suffix}.npy')
        alphastar_N = np.mean(_alphastar_N, axis=1)
        axin.plot(
            H_arr, alphastar_N, color='firebrick', marker='o', mfc='white', markersize=2.5,
            markevery=2, linestyle='--', linewidth=0.75
        )
        _alphastar_R = np.load(_rdir+f'alphastar_R{suffix}.npy')
        alphastar_R = np.mean(_alphastar_R, axis=1)
        axin.plot(
            H_arr, alphastar_R, color='k', marker='s', mfc='white', markersize=2.5,
            markevery=2, linestyle='-', linewidth=0.75
        )
        # Limits, labels, etc
        axin.set_xlim(0,1)
        axin.set_ylim(1,1.6)
        axin.set_ylabel(r'$\alpha^*$', fontsize=11)
        axin.set_yticks([1,1.2,1.4,1.6])
        axin.set_xticks([0,0.2,0.4,0.6,0.8,1.])
        axin.xaxis.set_minor_locator(MultipleLocator(0.1))
        axin.set_xticklabels([0,'','','','',1])
        axin.tick_params(axis='both', labelsize=8, length=2)
        axin.text(0.5,-0.105, r'$H$', fontsize=10, ha='center', va='top', transform=axin.transAxes)
        axin.yaxis.set_minor_locator(MultipleLocator(0.1))
        # fig.subplots_adjust(hspace=0.15)
        # Save
        self.figdict['effective_habitat_optimal_response'] = fig 
        

if __name__ == "__main__":
    Argus = src.args.Args() 
    args = Argus.args 
    Pjotr = Plotter()

    ## Lattice related plots
    # Pjotr.plot_lattice(args)
    # Pjotr.plot_predator_positions(args)
    # Pjotr.plot_lattice_dynamics(args)
    Pjotr.plot_fragmented_lattice(args)
    # Pjotr.plot_patch_distribution(args)  
    # Pjotr.plot_patch_percolation(args)
    # Pjotr.plot_lattice_habitat_loss_static(args)
    # Pjotr.plot_lattice_habitat_loss(args)

    ## Population density related plots
    # Pjotr.plot_population_densities_fragile(args)
    # Pjotr.plot_population_dynamics(args)
    # Pjotr.plot_population_densities(args)
    # Pjotr.plot_population_densities_alpha(args)
    # Pjotr.plot_population_densities_lambda(args)
    # Pjotr.plot_population_densities_sigma(args)
    # Pjotr.plot_population_densities_H(args)
    # Pjotr.plot_population_phase_space(args)
    # Pjotr.plot_preferred_fragmentation(args)
    # Pjotr.plot_species_richness(args)

    ## Flight length related plots
    # Pjotr.plot_flight_distribution_Lambda(args)

    ## Environmental related plots
    # Pjotr.plot_effective_habitat(args)
    # Pjotr.plot_habitat_loss_probability(args)
    # Pjotr.plot_heatmap(args)
    # Pjotr.plot_effective_habitat_optimal_response(args)
    
    if not args.save:
        plt.show()
    else:
        for figname, fig in Pjotr.figdict.items():
            print("Saving {}...".format(figname))
            fig.savefig(
                "figures/{}.pdf".format(figname), format='pdf', bbox_inches='tight', 
                pad_inches=0.01, transparent=True
            )
        