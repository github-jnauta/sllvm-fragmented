""" Callable for running stochastic Lotka-Volterra systems in fragmented landscapes """
# Import necessary libraries
import os, time 
import numpy as np 
# Import modules
import src.args 
import src.restricted_stochastic_lotka_volterra

if __name__ == "__main__":
    # Instantiate objects
    Argus = src.args.Args() 
    args = Argus.args 
    System = src.restricted_stochastic_lotka_volterra.SLLVM(args.seed)
    # Start clock for computation time estimates
    starttime = time.time()
    # Run
    output = System.run_system(args)

    if not args.nosave:
        # Specify directory, and make it if it does not yet exist
        _dir = args.ddir+'sllvm/{:s}/{L:d}x{L:d}/H{H:.4f}/'.format(
            args.argument, L=2**args.m, H=args.H
        )        
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        suffix = (
            '_T{:d}_N{:d}_M{:d}_H{:.4f}'
            '_rho{:.3f}_mu{:.4f}_Lambda{:.4f}_lambda{:.4f}_sigma{:.4f}_alpha{:.3f}'
            '_seed{:d}'.format(
                args.T, args.N0, args.M0, args.H, args.rho, 
                args.mu, args.Lambda_, args.lambda_, args.sigma, args.alpha,
                args.seed
            )
        )
        # Save
        if not args.nosave:
            for key, item in output.items():
                np.save(_dir + "{name:s}{suffix:s}".format(name=key, suffix=suffix), item)

    # Print some closing statements
    printstr = (
        '{L}x{L} lattice, \nH={H:.4f}, \u03C1={rho:.3f}, T={T:d}, \u03B1={alpha:.3f}, ' 
        '\u03BC={mu:.4f}, \u039B={Lambda_:.4f}, \u03BB={lambda_:.4f}, '
        '\u03C3={sigma:.4f}, seed {seed:d}'.format(
            L=2**args.m, H=args.H, rho=args.rho, T=args.T,
            alpha=args.alpha, mu=args.mu, Lambda_=args.Lambda_, 
            lambda_=args.lambda_, sigma=args.sigma, seed=args.seed
        )
    )
    seconds = time.time() - starttime
    minutes = seconds / 60 
    hours = minutes / 60 
    timestr = "%.4fs (%.2fmin) (%.2fhrs)"%(seconds, minutes, hours)
    print("Computations finished for %s\napprox. time: %s"%(printstr, timestr))



