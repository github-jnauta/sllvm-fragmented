""" Compute the box-dimension of the prey lattice """
# Import necessary libraries
import numpy as np 
import numba
from numpy.core.fromnumeric import mean 

###################
# Numba functions #
# @numba.jit(nopython=True, cache=True)
def compute_box_dimension(lattice):
    """ Compute the box-dimension of a binary lattice where 0s represent empty
        sites, and 1s occupied sites

        Parameters
        ----------
        lattice : np.array(L², dtype=np.int64)
            1-dimensional representation of the 2-dimensional lattice
            Note that the lattice has periodic boundary conditions, and thus the 
            box-counting method should account for this
    """
    L_sq = len(lattice)
    L = np.int64(np.sqrt(L_sq))
    _lattice = np.reshape(lattice, (L,L))
    # Specify the box sizes
    nsizes = np.int64(np.log2(L))
    _size = 1
    box_sizes = [_size*2**i for i in range(nsizes)]
    Dbox = np.zeros(len(box_sizes))
    # Compute box-dimension when box encompasses entire lattice
    Dbox[0] = np.sum(_lattice)
    # Loop through all the different box sizes and apply the box-counting method
    for i in range(nsizes-1, 0, -1):
        # Since we have periodic boundary conditions, just start at (0,0)
        idx = 0 
        ## Compute the number of filled lattice sites within the box at each offset
        # The number of offsets (corner of the box) is equal to the 2*box_size - 1
        delta = [[0,n] for n in range(box_sizes[i])] + [[n,0] for n in range(1,box_sizes[i])]
        noffsets = 2*box_sizes[i] - 1
        N = np.zeros(noffsets)
        for j in range(noffsets):
            # Roll the lattice
            axis = np.argmax(delta[j])
            temp_lattice = np.roll(_lattice, delta[j][axis], axis=axis)
            # Reshape the lattice to be able to take the mean over appropriate axes
            shape = (L//box_sizes[i], box_sizes[i], L//box_sizes[i], box_sizes[i])
            temp_lattice = np.reshape(_lattice, shape)
            # Compute the mean
            mean_lattice = np.mean(temp_lattice, axis=(1,-1))
            # Convert back to a binary lattice as all values below 0.5 are 0, above are 1
            binary_mean_lattice = np.floor(2*mean_lattice)
            # Compute number of filled boxes
            N[j] = np.sum(binary_mean_lattice)
        
        Dbox[i] = np.mean(N) 
    return Dbox, box_sizes

if __name__ == "__main__":
    # Construct an example lattice
    L = 2**9
    lattice = np.random.randint(2, size=L*L, dtype=np.int64)
    Dbox, box_sizes = compute_box_dimension(lattice)
