""" Analyze raw data """
# Import necessary libraries
import os
import numpy as np 
# Import modules
import src.args 

class Analyzer():
    def __init__(self):
        pass 

    def compute_population_densities(self, args):
        """ Compute the average population in the quasistationary state """ 
        # Specify and/or crease directories
        _dir = args.ddir+"sllvm/{L:d}x{L:d}/".format(L=2**args.m)
        _rdir = args.rdir+"sllvm/{L:d}x{L:d}/".format(L=2**args.m)
        if not os.path.exists(_rdir):
            os.makedirs(_rdir)
        # Load variable arrays
        lambda_arr = np.loadtxt(_dir+"lambda.txt")
        seeds = np.loadtxt(_dir+"seeds.txt", dtype=int)
        # Allocate
        N = np.zeros((len(lambda_arr), len(seeds)))
        M = np.zeros((len(lambda_arr), len(seeds)))
        for i, lambda_ in enumerate(lambda_arr):
            for j, seed in enumerate(seeds):
                suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}_lambda{:.4f}_sig{:.4f}_a{:.3f}_seed{:d}".format(
                    args.T, args.N0, args.M0, args.H, args.rho, 
                    args.mu, lambda_, args.sigma, args.alpha, seed
                )
                _N = np.load(_dir+"pred_population{suffix:s}.npy".format(suffix=suffix))
                _M = np.load(_dir+"prey_population{suffix:s}.npy".format(suffix=suffix))
                _Nmean = np.mean(_N[-5:], axis=0)
                _Mmean = np.mean(_M[-5:], axis=0)
                N[i,j] = _Nmean
                M[i,j] = _Mmean
        # Save
        save_suffix = "_T{:d}_N{:d}_M{:d}_H{:.3f}_rho{:.3f}_mu{:.4f}_sig{:.4f}_a{:.3f}".format(
            args.T, args.N0, args.M0, args.H, args.rho, 
            args.mu, args.sigma, args.alpha
        )
        np.save(_rdir+"N{suffix:s}".format(suffix=save_suffix), N)
        np.save(_rdir+"M{suffix:s}".format(suffix=save_suffix), M)
        # Print closing statements
        printstr = "{L}x{L} lattice, H={H:.3f}, \u03C1={rho:.3f}, T={T:d}, \u03B1={alpha:.3f}, \u03BC={mu:.4f}, \u03C3={sigma:.4f}".format(
            L=2**args.m, H=args.H, rho=args.rho, T=args.T,
            alpha=args.alpha, mu=args.mu, lambda_=args.lambda_, sigma=args.sigma, seed=args.seed
        )
        print("Computed quasistationary population densities for %s"%(printstr))
    

if __name__ == "__main__":
    # Instantiate class objects
    Argus = src.args.Args()
    args = Argus.args 
    Analyze = Analyzer() 
    # Analyze
    Analyze.compute_population_densities(args)
    

    
        
