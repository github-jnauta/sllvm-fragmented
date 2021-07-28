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
    # Specify bins for distribution computation
    bins = np.logspace(0, np.log10(args.rho*L**2+1), num=args.nbins, dtype=np.int64)
    bins = np.unique(bins)
    total_patch_number = 0 
    pdf = np.zeros(len(bins))

    # Compute the average patch size and number of patches
    for k in range(args.nmeasures):
        # Compute fractional Brownian surface
        fBs = Lattice.SpectralSynthesis2D(L, args.H)
        # Compute binary lattice
        lattice = Lattice.binary_lattice(fBs, args.rho)
        # Compute the labelled lattice using periodic boundary conditions
        labelled_lattice, num_labels = src.lattice.nb_applyPBC(*label(lattice))
        # Find the label for which lattice entries are empty        
        labels, sizes = np.unique(labelled_lattice, return_counts=True)
        for lab in labels:
            if not np.any(lattice[np.where(labelled_lattice==lab)]):
                break
        mask = np.ones(len(labels), bool)
        mask[np.argwhere(labels==lab)] = False
        # Apply the mask
        labels = labels[mask]
        sizes = sizes[mask]
        # Determine frequency
        indices = np.searchsorted(bins, sizes)
        indices, counts = np.unique(indices, return_counts=True)
        pdf[indices] += counts
        # Compute the max patch size
        patch_size[k] = np.max(sizes)
        num_patches[k] = num_labels
        total_patch_number += num_patches[k]
    # Normalize frequency to compute pdf
    pdf = pdf / total_patch_number
    # Save 
    suffix = '_H{:.3f}_rho{:.3f}'.format(args.H, args.rho)
    _dir = f'../data/patch_distribution/{L}x{L}/'
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    # np.save(_dir+f'patch_size{suffix}', patch_size)
    np.save(_dir+f'patch_distribution{suffix}', pdf)
    # np.save(_dir+f'num_patches{suffix}', num_patches)
    # Print closing statement
    print(f'Computed for H={args.H}, \u03C1={args.rho}')