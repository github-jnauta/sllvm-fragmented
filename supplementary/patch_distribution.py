""" Code for numerical approach to patch size and number distribution for different
    values of the Hurst exponent H and the occupation level ρ
"""
# Import necessary libraries
import sys, os
import numpy as np 
from scipy.ndimage import label
sys.path.append("../")
# Import modules
import src.args
import src.lattice


if __name__ == "__main__":
    # Extract arguments
    Argus = src.args.Args()
    args = Argus.args
    # Initialize additional objects
    Lattice = src.lattice.Lattice(args.seed)

    # Specify some variables
    L = 2**args.m
    # Allocate
    patch_size = np.zeros(args.nmeasures)
    num_patches = np.zeros(args.nmeasures)

    # Compute the average patch size and number of patches
    for k in range(args.nmeasures):
        # Compute fractional Brownian surface
        fBs = Lattice.SpectralSynthesis2D(L, args.H)
        # Compute binary lattice
        lattice = Lattice.binary_lattice(fBs, args.rho)
        # Compute the labelled lattice using periodic boundary conditions
        labelled_lattice, num_labels = src.lattice.nb_applyPBC(*label(lattice))
        # Compute the mean patch size
        labels, sizes = np.unique(labelled_lattice, return_counts=True)
        patch_size[k] = np.mean(sizes)
        num_patches[k] = num_labels
    
    # Save 
    suffix = '_H{:.3f}_rho{:.3f}'.format(args.H, args.rho)
    _dir = f'../data/patch_distribution/{L}x{L}/'
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    np.save(_dir+f'patch_size{suffix}', patch_size)
    np.save(_dir+f'num_patches{suffix}', num_patches)