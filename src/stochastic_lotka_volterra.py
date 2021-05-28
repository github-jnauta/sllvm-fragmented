""" Contains modules for executing Monte Carlo simulation of the stochastic
    Lotka-Volterra system. A single Monte Carlo step corresponds to selecting
    all individuals (cells/sites) on average, i.e. a single Monte Carlo step
    corresponds to randomly selecting L*L sites and evolving them accordingly.
    Foragers undergo (cardinal) Levy walks with constant velocity equal to the
    difference between lattice sites, normalized to 1. 

    Most modules are implemented using numba for speed purposes. Numba modules 
    are identified by the header @numba.jit(), and are cached (cache=True). Hence,
    while the initial run wherein these modules need to be compiled might be a bit
    slow(er), subsequent calls should be much faster.
"""
# Import necessary libraries
import numpy as np 
import numba 
# Import modules
import src.lattice

###################
# Numba functions #
# --------------- #
# General functions
@numba.jit(nopython=True, cache=True)
def nb_set_seed(seed):
    np.random.seed(seed)

@numba.jit(nopython=True, cache=True)
def nb_sample_cdf(cdf):
    """ Generate sample from given (discrete) cdf """
    rand = np.random.random() 
    for i in range(len(cdf)):
        if rand < cdf[i]:
            return i + 1            

# Specific functions
@numba.jit(nopython=True, cache=True)
def nb_truncated_zipf(alpha, N):
    """ Compute the cumulative distribution function for the truncated
        discrete zeta distribution (Zipf's law)
    """
    x = np.arange(1, N+1)
    weights = x ** -alpha
    weights /= np.sum(weights)
    cdf = np.cumsum(weights)
    return cdf 

@numba.jit(nopython=True, cache=True)
def nb_get_1D_neighbors(idx, L):
    """ Get the 1D neighbors of a specific index, abiding periodic boundary conditions
        on the (original) 2D lattice
    """
    # Convert index to 2D index
    i, j = idx // L, idx % L 
    neighbors = np.array([
        [(i+1)%L, j], [(i-1)%L,j], [i,(j+1)%L], [i,(j-1)%L]
    ], dtype=np.int64)
    neighbors_1D = np.array([n[0]*L + n[1] for n in neighbors], dtype=np.int64)
    return neighbors_1D

@numba.jit(nopython=True, cache=True)
def nb_SLLVM(T, N0, M0, sites, mu, lambda_, sigma, alpha, nmeasures):
    """ Runs the stochastic lattice Lotka-Volterra model
        While the lattice is a 2D (square) lattice, for speed we first convert everyting
        to a 1D lattice, where neighbors and periodic boundary conditions properly need
        to be taken into account.

        Parameters
        ----------
        T : np.int64
            The number of Monte Carlo time steps
        N0 : np.int64
            The initial number of predators (foragers)
        M0 : np.int64
            The initial number of prey (resources)
            Note that M0=min(M0,ρL²), where ρL² the number of sites eligible for prey
            If M0 = -1, then all eligible sites are initially filled with prey
        sites : np.array((L,L),dtype=np.int64)
            L x L numpy array of eligible sites for prey (1) and empty sites (0)
        mu : np.float64
            Mortality rate of the predators (foragers)
        lambda_ : np.float64
            Reproduction rate of the predators
        sigma : np.float64
            Reproduction rate of the prey (resources)
        alpha : np.float64
            Levy walk parameter of the predators' movement
        nmeasures : np.int64
            Number of times populations are measures at equally spaced intervals
    """
    L, _ = sites.shape 
    dmeas = T // nmeasures
    ## Initialize constants
    delta_idx = np.array([1, -1, L, -L], dtype=np.int64)
    delta_idx_2D = np.array([[0,1], [0,-1], [1,0], [-1,0]], dtype=np.int64)
    # Compute cdf for Zipf's law as the discrete (truncated) power law distribution
    _cdf = nb_truncated_zipf(alpha, L)
    ## Convert the eligible sites to 1D array
    sites = sites.flatten()
    ## Initialize the 1D lattice
    #  Each site on the lattice can contain three states:
    #   -1  : prey (resource)
    #   >=1 : predator (forager) [multiple occupancy is allowed (for now)]
    lattice = np.zeros(L*L, dtype=np.int64)
    # Initialize predators on sites not eligible for prey
    empty_sites = np.where(sites==0)[0]
    predator_idxs = np.random.choice(empty_sites, size=N0, replace=False)
    for i in predator_idxs:
        lattice[i] = 1
    # Initialize prey on eligible sites
    eligible_sites = np.where(sites==1)[0]
    M0 = min(M0, len(eligible_sites))
    M0 = len(eligible_sites) if M0 == -1 else M0
    prey_idxs = np.random.choice(eligible_sites, size=M0, replace=False)
    for i in prey_idxs:
        lattice[i] = -1   
    ## Compute occupied site
    occupied_sites = np.where(lattice!=0)[0]
    ## Allocate
    flight_length = np.zeros(N0, dtype=np.int64)    # Sampled flight length of each predator
    curr_length = np.zeros(N0, dtype=np.int64)      # Current length of each predator
    didx = np.zeros(N0, dtype=np.int64)             # 1D Δi for each predator
    predator_ids = np.arange(N0, dtype=np.int64)    # ID of each predator
    current_max_id = N0 - 1
    # Get predator positions seperately as is convenient for emulating Levy walks
    predator_pos = np.asarray([i for i in predator_idxs], dtype=np.int64)
    # Initialize number of predators and prey
    N = N0 
    M = M0 
    K = N0 + M0

    ## Allocate arrays for storing measures
    prey_population = np.zeros(nmeasures+1, dtype=np.int64)
    pred_population = np.zeros(nmeasures+1, dtype=np.int64)
    coexistence = 1
    # lattice_configuration = np.zeros((L*L, nmeasures), dtype=np.int64)
    # predator_positions = np.zeros((N0,T), dtype=np.int64)
    # Store initial values
    prey_population[0] = M0 
    pred_population[0] = N0 

    ## Run the stochastic Lotka-Volterra system
    for t in range(1,T+1):
        if t % dmeas == 0:
            # Store desired variables
            imeas = t // dmeas
            prey_population[imeas] = M 
            pred_population[imeas] = N
            # lattice_configuration[:,imeas] = lattice.copy()
        # Stop the simulation if either of the populations has become extinct
        if M == 0 or N==0:
            coexistence = 0
            break
        # Set the fixed number of sites to be evolved
        # A single loop that selects (on average) each occupied site once is considered
        # a single Monte Carlo time step
        steps = K 
        for tau in range(steps):
            # Select a random occupied site
            _k = np.random.randint(0, K)
            idx = occupied_sites[_k]
            neighbors = nb_get_1D_neighbors(idx, L)
            ## If the site contains prey, reproduce with probability (rate) σ
            #NOTE: Prey only reproduces if it has an empty eligible neighboring site
            if lattice[idx] == -1:
                # Check if neighboring sites are eligible
                eligible_neighbors = [n for n in neighbors if sites[n] and not lattice[n]]
                _nn = len(eligible_neighbors)
                if _nn > 0:
                    # Randomly sample one of the eligible neighboring sites
                    neighbor = eligible_neighbors[np.random.randint(0,_nn)]
                    # Place prey there with probability σ
                    if np.random.random() < sigma:
                        lattice[neighbor] = -1
                        M += 1
                        occupied_sites = np.append(occupied_sites, neighbor)
                        K += 1
            ## If the site contains a predator, check in order
            # (i)   die with mortality rate μ
            # (ii)  start a new flight
            # (iii) continue the current flight
            # (iv)  consume prey and reproduce
            elif lattice[idx] > 0:
                ## Get predator ID. If two on the same location, select one
                for _pred_id, _pred_idx in enumerate(predator_pos):
                    if _pred_idx == idx:
                        break
                ## (i) Die with mortality rate μ
                #NOTE: We can substract 1, such that if the current state is the reproductive
                #      state with value 2, there is 1 predator still remaining
                if np.random.random() < mu:
                    lattice[idx] -= 1
                    # Remove the predator
                    # predator_ids = predator_ids[np.nonzero(predator_ids!=_pred_id)]
                    # predator_pos = predator_pos[np.nonzero(predator_ids!=_pred_id)]
                    # curr_length = curr_length[np.nonzero(predator_ids!=_pred_id)]
                    predator_ids = np.delete(predator_ids, _pred_id)
                    predator_pos = np.delete(predator_pos, _pred_id)
                    flight_length = np.delete(flight_length, _pred_id)
                    curr_length = np.delete(curr_length, _pred_id)
                    N = max(0, N-1)
                    occupied_sites = np.delete(occupied_sites, _k)
                    K = max(0, K-1)
                else:
                    ## (ii) start a new flight
                    if curr_length[_pred_id] == 0:
                        flight_length[_pred_id] = nb_sample_cdf(_cdf)
                        didx[_pred_id] = np.random.randint(0,4)
                    
                    ## (iii) continue the current (or just started) flight
                    # Update position
                    # NOTE: The lattice is updated below (iv)
                    lattice[idx] -= 1
                    # Determine new 1D index using periodic boundary conditions
                    _i = ( idx // L + delta_idx_2D[didx[_pred_id]][0] ) % L
                    _j = ( idx % L + delta_idx_2D[didx[_pred_id]][1] ) % L 
                    new_idx = _i * L + _j
                    # Ensure single occupancy
                    if lattice[new_idx] > 0:
                        new_idx = idx                   # Stay put
                        curr_length[_pred_id] = 0       # Truncate current flight
                    else:                        
                        curr_length[_pred_id] += 1      # Increment current path length
                    # Update predator position
                    predator_pos[_pred_id] = new_idx
                    # End current flight if path length exceeds the sampled length
                    if curr_length[_pred_id] > flight_length[_pred_id]:
                        curr_length[_pred_id] = 0 

                    if lattice[new_idx] == -1:
                        ## (iv) consume prey and reproduce
                        lattice[new_idx] = 1            # Site becomes occupied by predator
                        curr_length[_pred_id] = 0       # Truncate current flight
                        occupied_sites[_k] = new_idx    # Update index occupied by predator
                        ## Consume prey and remove occupied site
                        M = max(0, M-1)
                        for __k, __idx in enumerate(occupied_sites):
                            if __idx == new_idx:
                                break
                        occupied_sites = np.delete(occupied_sites, __k)
                        K = max(0, K-1)
                        ## Reproduce with rate λ
                        if np.random.random() < lambda_:
                            lattice[new_idx] += 1
                            predator_ids = np.append(predator_ids, current_max_id)
                            predator_pos = np.append(predator_pos, new_idx)
                            flight_length = np.append(flight_length, 0)
                            curr_length = np.append(curr_length, 0)
                            didx = np.append(didx, 0)
                            current_max_id += 1
                            N += 1
                            # Add occupied site
                            occupied_sites = np.append(occupied_sites, new_idx)
                            K += 1
                    else:
                        # NOTE: Update lattice as mentioned above
                        # No prey, so predator just moves there
                        lattice[new_idx] += 1
                        # Update site index occupied by the predator
                        occupied_sites[_k] = new_idx
    
    return prey_population, pred_population, coexistence

#################################
# Wrapper for the numba modules #
class SLLVM(object):
    """ Class for the Stochastic Lattice Lotka-Volterra Model """
    def __init__(self) -> None: 
        self.Lattice = src.lattice.Lattice()

    def run_system(self, args):
        # Fix the RNG
        np.random.seed(args.seed)
        nb_set_seed(args.seed)
        # Compute the prey (resource) sites on the L x L lattice
        _lattice = self.Lattice.SpectralSynthesis2D(2**args.m, args.H)
        sites = self.Lattice.binary_lattice(_lattice, args.rho)
        # Initialize dictionary
        outdict = {}
        outdict['prey_population'] = np.zeros((args.nmeasures+1, args.reps), dtype=np.int64)
        outdict['pred_population'] = np.zeros((args.nmeasures+1, args.reps), dtype=np.int64)
        outdict['coexistence'] = np.zeros(args.reps, dtype=np.int64)
        # Repeat the SLLVM for the lattice and gather results
        for rep in range(args.reps):
            output = nb_SLLVM(
                args.T, args.N0, args.M0, sites, 
                args.mu, args.lambda_, args.sigma, args.alpha,
                args.nmeasures
            )
            outdict['prey_population'][:,rep] = output[0]
            outdict['pred_population'][:,rep] = output[1]
            outdict['coexistence'][rep] = output[2]
        return outdict

    