""" Analyze raw data """
# Import necessary libraries
import os
import numpy as np 
# Import modules
import src.args 

class Analyzer():
    def __init__(self):
        pass 

    def _get_string_dependent_vars(self, args):
        """ Select the argument to compute against
            i.e. x is the argument where f(x) is computed against
        """
        # Compute some variables
        L = 2**args.m
        # Determine directory inputs
        self._dir = args.ddir+'sllvm/{arg:s}/{L:d}x{L:d}/'.format(arg=args.argument, L=L)
        self._rdir = args.rdir+'sllvm/{name:s}/{L:d}x{L:d}/'.format(name=args.argument, L=L)
        # Make directory if it does not exist
        if not os.path.exists(self._rdir):
            os.makedirs(self._rdir)
        # Load specific variable (argument) array
        self._var_arr = np.loadtxt(self._dir+'{name:s}.txt'.format(name=args.argument))
        # Specify suffix depending on the argument 
        # (adapt as necessary)
        if args.argument == 'lambda':
            self._suffix = (
                '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}'
                '_Lambda{:.4f}_lambda{:s}_sig{:.4f}_a{:.3f}_seed{:s}'.format(
                    args.T, args.N0, args.M0, args.H, args.rho, args.mu,
                    args.Lambda_, '{var:.4f}', args.sigma, args.alpha, '{seed:d}'
                )
            )
            self._printstr = (
                '{L}x{L} lattice, H={H:.3f}, \u03C1={rho:.3f}, T={T:d}, ' \
                '\u039B={Lambda_:.4f}, \u03B1={alpha:.3f}, ' \
                '\u03BC={mu:.4f}, \u03C3={sigma:.4f}'.format(
                    L=2**args.m, H=args.H, rho=args.rho, T=args.T,
                    Lambda_=args.Lambda_, alpha=args.alpha,
                    mu=args.mu, sigma=args.sigma
                )
            )
            self.save_suffix = '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}' \
                'Lambda{:.4f}_alpha{:.3f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, args.H,
                args.rho, args.Lambda_, args.alpha, args.mu, args.sigma
            )
        elif args.argument == 'alpha':
            self._suffix = (
                '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}'
                '_Lambda{:.4f}_lambda{:.4f}_sig{:.4f}_a{:s}_seed{:s}'.format(
                    args.T, args.N0, args.M0, args.H, args.rho, args.mu,
                    args.Lambda_, args.lambda_, args.sigma, '{var:.3f}', '{seed:d}'
                )
            )
            self._printstr = (
                '{L}x{L} lattice, H={H:.3f}, \u03C1={rho:.3f}, T={T:d}, ' \
                '\u039B={Lambda_:.4f}, \u03BB={lambda_:.4f}, ' \
                '\u03BC={mu:.4f}, \u03C3={sigma:.4f}'.format(
                    L=2**args.m, H=args.H, rho=args.rho, T=args.T,
                    Lambda_=args.Lambda_, lambda_=args.lambda_, 
                    mu=args.mu, sigma=args.sigma
                )
            )
            self.save_suffix = '_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}' \
                'Lambda{:.4f}_lambda{:.4f}_mu{:.4f}_sigma{:.4f}'.format(
                args.T, args.N0, args.M0, args.H, 
                args.rho, args.Lambda_, args.lambda_, args.mu, args.sigma
            )
        else:
            print('No specified suffix structure for given argument: {:s}'.format(args.argument))
            exit()
        

    def compute_population_densities(self, args):
        """ Compute the average population in the quasistationary state """ 
        # Get directories based on the argument
        self._get_string_dependent_vars(args)
        # Load variable arrays
        seeds = np.loadtxt(self._dir+"seeds.txt", dtype=int)
        # Allocate
        N = np.zeros((len(self._var_arr), len(seeds)))
        M = np.zeros((len(self._var_arr), len(seeds)))
        for i, var in enumerate(self._var_arr):
            for j, seed in enumerate(seeds):
                suffix = self._suffix.format(var=var, seed=seed)
                _N = np.load(self._dir+"pred_population{suffix:s}.npy".format(suffix=suffix))
                _M = np.load(self._dir+"prey_population{suffix:s}.npy".format(suffix=suffix))
                N[i,j] = np.mean(_N[-25:])
                M[i,j] = np.mean(_M[-25:])
        # Save
        np.save(self._rdir+"N{suffix:s}".format(suffix=self.save_suffix), N)
        np.save(self._rdir+"M{suffix:s}".format(suffix=self.save_suffix), M)
        # Print closing statements
        print("Computed quasistationary population densities for \n %s"%(self._printstr))
    

if __name__ == "__main__":
    # Instantiate class objects
    Argus = src.args.Args()
    args = Argus.args 
    Analyze = Analyzer() 
    # Analyze
    Analyze.compute_population_densities(args)
    

    
        
