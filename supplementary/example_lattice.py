""" Code for generating binary fragmented lattices using fractional
    Brownian motion with Hurst exponent H and occupancy level ρ
"""
# Import necessary libraries
import numpy as np 
import sys 
sys.path.insert(0, '../')
# Import modules
import src.args 
import src.lattice 

if __name__ == "__main__":
    Argus = src.args.Args() 
    args = Argus.args 
    Lattice = src.lattice.Lattice(args.seed)
    # Generate lattice
    _lattice = Lattice.SpectralSynthesis2D(2**args.m, args.H)
    lattice = Lattice.binary_lattice(_lattice, args.rho)
    # Save
    suffix = "_{L:d}x{L:d}_H{H:.3f}_rho{rho:.3f}".format(
        L=2**args.m, H=args.H, rho=args.rho
    )
    np.save("../data/landscapes/lattice{suffix:s}".format(suffix=suffix), lattice)

