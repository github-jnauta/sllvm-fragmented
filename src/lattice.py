""" Holds modules for generating resource lattices using fractional Brownian motion
    Currently, the main function that generates an L x L landscape uses the 
    two-dimensional spectral synthesis from (Saupe, 1988, Algorithms for random fractals).
"""
# Import necessary libraries
import numpy as np 
import numba
from numba.core import types
from numba.typed import Dict
from scipy.ndimage import label
# Define the type used in the dictionary
int_ = types.int64
int_array = types.int64[:]

@numba.jit(nopython=True, cache=True)
def nb_set_seed(seed):
    np.random.seed(seed)

@numba.jit(nopython=True, cache=True)
def nb_SpectralSynthesis2D(L, H, sig):
    A = np.zeros((L,L), dtype=np.cdouble)
    for i in range(L//2):
        for j in range(L//2):
            phase = 2*np.pi*np.random.random()
            if i!=0 or j!=0:
                r = (i*i+j*j)**(-(H+1)/2) * np.random.normal(0, sig)
            else:
                r = 0 
            A[i,j] = r*np.cos(phase) + 1j*r*np.sin(phase)
            i0 = 0 if i == 0 else L-i 
            j0 = 0 if j == 0 else L-j 
            A[i0,j0] = r*np.cos(phase) + 1j*r*np.sin(phase)

    # @TODO: Why does one need to loop 'twice'
    # (but note different indices are assigned)
    # See also https://link.springer.com/content/pdf/10.1023/A:1008193015770.pdf 
    for i in range(1, L//2):
        for j in range(1, L//2):
            phase = 2*np.pi*np.random.random() 
            r = (i*i+j*j)**(-(H+1)/2) * np.random.normal(sig)
            A[i,L-j] = r*np.cos(phase) + 1j*r*np.sin(phase)
            A[L-i,j] = r*np.cos(phase) - 1j*r*np.sin(phase)
    return A

@numba.jit(nopython=True, cache=True)
def nb_applyPBC(lattice, num_labels):
    """ Apply periodic boundary conditions on a labelled lattice """
    L, _ = lattice.shape
    for i in range(L):
        # Compare left and right        
        if lattice[i,0]==1 and lattice[i,-1]==1:
            if lattice[i,0] != lattice[i,-1]:
                lattice = np.where(lattice==lattice[i,-1], lattice[i,0], lattice)
                num_labels -= 1
        # Compare up and down
        if lattice[0,i] and lattice[-1,i]:
            if lattice[0,i] != lattice[-1,i]:
                lattice = np.where(lattice==lattice[-1,i], lattice[0,i], lattice)
                num_labels -= 1
    return lattice, num_labels

@numba.jit(nopython=True, cache=True)
def nb_construct_label_list(labelled_lattice, labels):
    # Go through each label, and gather their indices and put them in a list of lists
    label_dict = Dict.empty(
        key_type=int_,
        value_type=int_array,
    )  
    for lab in labels:
        indices = np.flatnonzero(labelled_lattice==lab)
        label_dict[lab] = indices
    return label_dict

class Lattice(object):
    def __init__(self, seed) -> None:
        nb_set_seed(seed)

    def binary_lattice(self, lattice, rho):
        """ Generate binary lattice from the continuous fraction Brownian motion lattice 
            ρ ∈ [0,1] determines the occupancy, where ρN is the (integer) number of sites
            available to be occupied by resources
        """
        if rho == 1:
            return np.ones(lattice.shape, dtype=np.bool_)
        shifted_lattice = lattice + abs(np.min(lattice))    # Shift
        sorted_lattice = np.sort(shifted_lattice.flatten()) # Sort 
        # Determine cutoff point
        cutoff = sorted_lattice[int((1-rho)*lattice.shape[0]*lattice.shape[1])]
        # Generate binary lattice
        _lattice = shifted_lattice / cutoff                 # Normalize lattice
        _lattice[_lattice >= 1] = 1                         # All above cutoff to 1
        _lattice[_lattice < 1] = 0                          # All below to 0
        return np.asarray(_lattice, dtype=np.bool_)
    
    def label(self, lattice):
        """ Compute the labelled lattice using periodic boundary conditions """
        labelled_lattice, num_labels = nb_applyPBC(*label(lattice))              
        labels = np.unique(labelled_lattice)
        # Find the label for which lattice entries are empty
        for lab in labels:
            if not np.any(lattice[np.where(labelled_lattice==lab)]):
                break
        # Compute mask for those indices
        mask = np.ones(len(labels), bool)
        mask[np.argwhere(labels==lab)] = False
        # Apply the mask
        labels = labels[mask]
        label_list = nb_construct_label_list(labelled_lattice, labels)        
        return label_list, num_labels

    def SpectralSynthesis2D(self, L, H, sig=1, bounds=[0,1]):
        """ Generate fractional Brownian in two dimensions 
            @TODO: Fully understand the algorithm below, and clarify with comments
        """
        A = nb_SpectralSynthesis2D(L, H, sig)
        X = np.real(np.fft.fft2(A))
        return X 
