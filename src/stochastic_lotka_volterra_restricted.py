""" Contains modules for executing Monte Carlo simulation of the stochastic
    Lotka-Volterra system. A single Monte Carlo step corresponds to selecting
    all individuals (cells/sites) on average, i.e. a single Monte Carlo step
    corresponds to randomly selecting L*L sites and evolving them accordingly.
    Foragers undergo (cardinal) Levy walks with constant velocity equal to the
    difference between lattice sites, normalized to 1. 
    Prey is initially distributed on lattice sites generated by binary selection
    of two-dimensional fractional Brownian motion. Predators are initially
    distributed uniformly on empty sites. 

    Most modules are implemented using Numba for speed purposes. Numba modules 
    are identified by the header @numba.jit(nopython=True), and they are cached 
    (cache=True). Hence, while the initial run wherein these modules need to be 
    compiled might be a bit slow(er), subsequent calls should be much faster.
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
    """ Specifically call the seed from Numba code, as calling numpy.random.seed()
        from non-Numba code (or from object mode code) will seed the Numpy random 
        generator, not the Numba random generator.
    """
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
    # Adapt some variables as they should take on a specific value if -1 is provided
    mu = 1 / L if mu == -1 else mu          # Death rate 
    N0 = L**2 // 5 if N0 == -1 else N0     # Initial number of predators

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
    # Initialize prey on eligible sites
    prey_sites = np.where(sites==1)[0]
    M0 = min(M0, len(prey_sites))
    if M0 == -1:
        M0 = N0 if len(prey_sites)==L**2 else len(prey_sites)
    # M0 = len(prey_sites) if M0 == -1 else M0
    prey_idxs = np.random.choice(prey_sites, size=M0, replace=False)
    for i in prey_idxs:
        lattice[i] = -1   
    # Initialize predators on sites not eligible for prey    
    predator_sites = np.where(lattice!=-1)[0]
    predator_idxs = np.random.choice(predator_sites, size=N0, replace=False)
    for i in predator_idxs:
        lattice[i] = 1
    ## Compute occupied site
    occupied_sites = np.where(lattice!=0)[0]
    ## Allocate
    flight_length = np.zeros(N0, dtype=np.int64)    # Sampled flight length of each predator
    curr_length = np.zeros(N0, dtype=np.int64)      # Current length of each predator
    didx = np.zeros(N0, dtype=np.int64)             # 1D Δi for each predator
    predator_ids = np.arange(N0, dtype=np.int64)    # ID of each predator
    predator_pos = np.asarray([i for i in predator_idxs], dtype=np.int64)
    current_max_id = N0 - 1
    # Specify masks
    alive_mask = np.ones(N0, dtype=np.bool_)
    occupied_mask = np.ones(len(occupied_sites), dtype=np.bool_)
    # Initialize number of predators and prey
    N = N0 
    M = M0 
    K = N0 + M0
    # Specify temporary loop variable that allows for dynamic value of K
    temp_K = K 

    ## Allocate arrays for storing measures
    prey_population = np.zeros(nmeasures+1, dtype=np.int64)
    pred_population = np.zeros(nmeasures+1, dtype=np.int64)
    coexistence = 1
    # lattice_configuration = np.zeros((L*L, nmeasures+1), dtype=np.int64)
    # predator_positions = np.zeros((N0,T), dtype=np.int64)
    # Store initial values
    prey_population[0] = M0 
    pred_population[0] = N0 
    # lattice_configuration[:,0] = lattice.copy()

    ## Run the stochastic Lotka-Volterra system
    for t in range(1,T+1):
        ## Store desired variables every dmeas timesteps
        if t % dmeas == 0:
            imeas = t // dmeas
            prey_population[imeas] = M 
            pred_population[imeas] = N
            # lattice_configuration[:,imeas] = lattice.copy()

        ## Stop the simulation if:
        # prey goes extinct, as predators will also go extinct
        if M == 0:
            coexistence = 0
            break
        # predators go extinct, as prey will fully occupy all sites
        if N == 0:
            coexistence = 0
            prey_population[imeas:] = L**2 
            break 

        ## Update using the masked arrays
        # Predator related arrays
        predator_ids = predator_ids[alive_mask]
        predator_pos = predator_pos[alive_mask]
        flight_length = flight_length[alive_mask]
        curr_length = curr_length[alive_mask]
        didx = didx[alive_mask]
        alive_mask = alive_mask[alive_mask]
        # Occupied sites
        occupied_sites = occupied_sites[occupied_mask]
        occupied_mask = occupied_mask[occupied_mask]
        # Set new temp value for K
        temp_K = K
        
        # Set the fixed number of sites to be evolved
        # A single loop that selects (on average) each occupied site once is considered
        # a single Monte Carlo time step
        for tau in range(temp_K):
            # Break out of the MC step if no site is occupied
            if K == 0:
                break 
            # Select a random occupied site
            site_is_occupied = False 
            while not site_is_occupied:                
                _k = np.random.randint(0, temp_K)
                site_is_occupied = occupied_mask[_k]
            idx = occupied_sites[_k]
            neighbors = nb_get_1D_neighbors(idx, L)
            ## If the site contains prey, reproduce with probability (rate) σ
            #NOTE: Prey only reproduces if it has an empty neighboring site
            if lattice[idx] == -1:
                # Check if neighboring sites are empty
                empty_neighbors = [n for n in neighbors if sites[n] and lattice[n]==0]
                _nempty = len(empty_neighbors)
                # Check if prey site is surrounded by other prey
                eligible_neighbors = [n for n in neighbors if sites[n] and lattice[n]!=-1]
                _neligible = len(eligible_neighbors)
                ## If there are empty neighboring sites, reproduce with rate σ
                if _nempty > 0:
                    # Randomly sample one of the eligible neighboring sites
                    neighbor = empty_neighbors[np.random.randint(0,_nempty)]
                    # Place prey there with probability σ
                    if np.random.random() < sigma:
                        lattice[neighbor] = -1
                        M += 1
                        occupied_sites = np.append(occupied_sites, neighbor)
                        occupied_mask = np.append(occupied_mask, True)
                        K += 1
                ## If it is surrounded, set the mask for the site to False
                # (NOTE that even though the site is still occupied, we should not take
                # it into account in our main loop as prey cannot reproduce)
                elif _neligible == 0:                    
                    occupied_mask[_k] = False
                    K -= 1
                else:
                    pass
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
                    # Change the alive_mask of the predator to False
                    alive_mask[_pred_id] = False 
                    N -= 1
                    # If previous site is now empty, set mask value to False
                    if lattice[idx] == 0:
                        occupied_mask[_k] = False 
                        K -= 1
                else:
                    ## (ii) start a new flight
                    if curr_length[_pred_id] == 0:
                        flight_length[_pred_id] = nb_sample_cdf(_cdf)
                        didx[_pred_id] = np.random.randint(0,4)
                    
                    ## (iii) continue the current (or just started) flight
                    # Determine new 1D index using periodic boundary conditions
                    _i = ( idx // L + delta_idx_2D[didx[_pred_id]][0] ) % L
                    _j = ( idx %  L + delta_idx_2D[didx[_pred_id]][1] ) % L 
                    new_idx = _i * L + _j
                    # Ensure single occupancy by doing nothing when the site is
                    # already occupied by a predator or predators
                    if lattice[new_idx] > 0:
                        curr_length[_pred_id] = 0       # Truncate current flight
                    else:
                        # Update the lattice by decrementing previously occupied site
                        lattice[idx] -= 1
                        # Ensure that, if the site was previously in the reproduction 
                        # state, that the increase in occupied sites is handled
                        if lattice[idx] == 1:
                            occupied_sites = np.append(occupied_sites, idx)
                            occupied_mask = np.append(occupied_mask, True)
                            K += 1
                        curr_length[_pred_id] += 1          # Increment current path length
                        predator_pos[_pred_id] = new_idx    # Update predator position
                        # End current flight if current path length exceeds sampled length
                        if curr_length[_pred_id] > flight_length[_pred_id]:
                            curr_length[_pred_id] = 0 

                        if lattice[new_idx] == -1:
                            ## (iv) consume prey and reproduce
                            lattice[new_idx] = 1        # Site becomes occupied by predator
                            curr_length[_pred_id] = 0   # Truncate current flight
                            M -= 1
                            # Ensure that site previously occupied by the predator has its
                            # mask value set to False, at it replaces the prey
                            occupied_mask[_k] = False
                            K -= 1
                            # If the occupied site mask was previously set to False when it
                            # was surrounded by other prey, set the mask of the site now 
                            # occupied by the predator to True
                            __k = np.argmax(occupied_sites==new_idx)
                            if __k == 0 and occupied_sites[__k] != new_idx:
                                occupied_sites = np.append(occupied_sites, new_idx)
                                occupied_mask = np.append(occupied_mask, True)
                                K += 1

                            ## Reproduce with rate λ
                            if np.random.random() < lambda_:
                                # Update the lattice
                                lattice[new_idx] += 1
                                # Append new predator to corresponding arrays
                                predator_ids = np.append(predator_ids, current_max_id)
                                predator_pos = np.append(predator_pos, new_idx)
                                flight_length = np.append(flight_length, 0)
                                curr_length = np.append(curr_length, 0)
                                didx = np.append(didx, 0)
                                alive_mask = np.append(alive_mask, True)
                                current_max_id += 1
                                N += 1
                        else:
                            ## Displace
                            lattice[new_idx] += 1
                            occupied_sites[_k] = new_idx
    return prey_population, pred_population, coexistence#, lattice_configuration

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
        if args.rho == 1:
            sites = np.ones((2**args.m, 2**args.m), dtype=np.int64)
        else:
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
        # outdict['lattice'] = output[3]
        # outdict['sites'] = sites
        return outdict

    
