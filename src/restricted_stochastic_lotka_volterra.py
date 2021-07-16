""" Contains modules for executing Monte Carlo simulation of the restricted stochastic
    Lotka-Volterra system. A single Monte Carlo step corresponds to selecting all 
    individuals (cells/sites) on average, i.e. a single Monte Carlo step corresponds 
    to randomly selecting K sites and evolving them accordingly. Here, K is the number
    of occupied sites at time t. Foragers undergo (cardinal) Levy walks with constant 
    velocity equal to the difference between lattice sites, normalized to 1. Prey is 
    initially distributed on lattice sites generated by binary selection of 
    two-dimensional fractional Brownian motion. Predators are initially distributed 
    uniformly on empty sites initially not occupied by prey. 

    Most modules are implemented using Numba for speed purposes. Numba modules 
    are identified by the header @numba.jit(nopython=True), and they are cached 
    (cache=True). Hence, while the initial run wherein these modules need to be 
    compiled might be a bit slow(er), subsequent calls should be much faster.
"""
# Import necessary libraries
from numba.core.errors import new_error_context
import numpy as np 
from collections import deque
import numba
from numpy.random import rand 
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

@numba.jit(nopython=True, cache=True)
def nb_sample_powlaw_discrete(alpha, xmin=1):
    """ Generate discrete sample from the (truncated) powerlaw distribution
        (see https://www.jstor.org/stable/pdf/25662336.pdf, Eq. D.6)

        Note: The used approximation is erronous for small values of xmin, and
              becomes more precise for larger values. In this particular case, we
              do not care so much about statistical precision, and hence we can 
              use the approximation for xmin=1
    """     
    _sample = (xmin-0.5)*(1-np.random.random())**(-1/(alpha-1)) + 0.5
    return np.int64(_sample)

@numba.jit(nopython=True, cache=True)
def nb_get_1D_neighbors(idx, L):
    """ Get the 1D neighbors of a specific index, abiding periodic boundary conditions
        on the (original) 2D lattice
    """
    # Convert index to 2D index
    i, j = idx // L, idx % L 
    neighbors = [[(i+1)%L, j], [(i-1)%L,j], [i,(j+1)%L], [i,(j-1)%L]]
    neighbors_1D = np.array([n[0]*L + n[1] for n in neighbors], dtype=np.int64)
    return neighbors_1D

# @profile
@numba.jit(nopython=True, cache=True)
def nb_SLLVM(
        T, N0, M0, sites, mu, lambda_, Lambda_, sigma, alpha, nmeasures, 
        visualize, xmin=1
    ):
    """ Runs the stochastic lattice Lotka-Volterra model
        While the lattice is a 2D (square) lattice, we first convert everyting to a 
        1D lattice, where neighbors and periodic boundary conditions properly need to 
        be taken into account. The reason is that it makes it easier to generate arrays
        of which the components are a single integer (1D index) instead of an array of
        two integers (2D index). 
        While most common Monte Carlo approaches to SLLVMs randomly select sites and act
        based on the type of site, this leads to some inefficiencies when either selecting
        empty sites or when selecting sites that cannot do anything, e.g. prey cannot 
        reproduce if they are surrounded by other prey. To alleviate this issue, we have to
        do some bookkeeping by generating a (boolean) lattice that holds True values for
        sites that should not be counted in the Monte Carlo time steps. One should be careful
        to reinclude them should they be counted again, e.g. when removing neighboring prey.

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
            Mortality rate of the predators (foragers) μ
        lambda_ : np.float64
            Reproduction rate of the predators λ, note that prey consumption rate is 
            then defined by 1-lambda_
        Lambda_ : np.float64
            Predator-prey interaction rate Λ
        sigma : np.float64
            Reproduction rate of the prey (resources) σ
        alpha : np.float64
            Levy walk parameter of the predators' movement α
        nmeasures : np.int64
            Number of times populations are measures at equally spaced intervals
        xmin : np.int64
            Minimum predator displacement. Default set to 1 as the lattice size.
        visualize : np.bool_
            Boolean flag that includes lattice configuration over time in return
    """
    # Specify some variables
    L, _ = sites.shape              # Size of LxL lattice
    dmeas = T // nmeasures          # Δt at which measurements need to be collected
    _nn = 4                         # Number of nearest neighbors
    # Adapt some variables as they should take on a specific value if -1 is provided
    mu = 1 / L if mu == -1 else mu              # Death rate 
    N0 = L**2 // 10 if N0 == -1 else N0         # Initial number of predators
    alpha = np.inf if alpha == -1 else alpha    # Levy parameter

    ## Initialize constants
    delta_idx_2D = [[0,1], [0,-1], [1,0], [-1,0]]

    ## Convert the eligible sites to 1D array
    sites = sites.flatten()
    ## Initialize the 1-dimensional lattices
    #  We need three lattices:
    #  * one lattice of integer type that contains the IDs of the predators
    #  * one lattice of boolean type that contains prey
    #  * one lattice of boolean type that contains sites not counted (see above)
    pred_lattice = np.zeros(L*L, dtype=np.int64)
    prey_lattice = np.zeros(L*L, dtype=np.bool_)
    not_counted_lattice = np.zeros(L*L, dtype=np.bool_)

    ## Distribute prey on eligible sites
    prey_sites = np.where(sites==1)[0]
    M0 = min(M0, len(prey_sites))
    if M0 == -1:
        M0 = 2*N0 if len(prey_sites)==L**2 else len(prey_sites)
    prey_idxs = np.random.choice(prey_sites, size=M0, replace=False)
    for i in prey_idxs:
        prey_lattice[i] = True 
    ## Distribute predators on sites that do not contain prey
    pred_sites = np.flatnonzero(~prey_lattice)
    pred_idxs = np.random.choice(pred_sites, size=N0, replace=False)
    for pred_idx, lattice_idx in enumerate(pred_idxs):
        # Predator lattice contains the IDs, and needs to be offset by 1 as to
        # not have ID 0, as 0 represents no predator on that position
        pred_lattice[lattice_idx] = pred_idx + 1

    ## Compute list that contains indices (positions) of occupied sites
    occupied_sites = [idx for idx in prey_idxs] + [idx for idx in pred_idxs]
    ## Allocate lists needed for individual predator behavior
    gen = range(N0)
    flight_length = [i for i in gen]    # Sampled flight length of individual predators
    curr_length = [0 for i in gen]      # Current path length of individual predators
    didx = [0 for i in gen]             # Current direction of individual predators
    current_max_id = N0 + 1

    ## Initialize
    N = N0          # Number of predators       
    M = M0          # Number of prey
    K = N0 + M0     # Number of occupied sites

    ## Allocate arrays for storing measures
    prey_population = np.zeros(nmeasures+1, dtype=np.int64)
    pred_population = np.zeros(nmeasures+1, dtype=np.int64)
    coexistence = 1  
    # Store initial values
    prey_population[0] = M0 
    pred_population[0] = N0 
    # Allocate and initialice lattice configuration if visualize flag is given
    lattice_configuration = np.zeros((L*L, nmeasures+1), dtype=np.int16)
    if visualize:
        lattice_configuration[prey_lattice,0] = -1 
        lattice_configuration[pred_lattice>0,0] = 1

    ##############################################
    ## Run the stochastic Lotka-Volterra system ##
    for t in range(1,T+1):
        ## Store desired variables every dmeas timesteps
        if t % dmeas == 0:
            imeas = t // dmeas
            prey_population[imeas] = M 
            pred_population[imeas] = N
            if visualize:
                lattice_configuration[prey_lattice,imeas] = -1 
                lattice_configuration[pred_lattice>0,imeas] = 1

        ## Stop the simulation if:
        # prey goes extinct, as predators will also go extinct
        if M == 0:
            coexistence = 0
            break
        # predators go extinct, as prey will fully occupy all available sites
        if N == 0:
            coexistence = 0
            prey_population[imeas:] = np.sum(sites)
            break 
        
        # Generate random numbers in bulk
        steps = K 
        randoms = np.random.random(steps)

        ## Evolve random sites according to possible transitions
        for tau in range(steps):
            # Select a random occupied site
            _k = np.int64(K*randoms[tau])
            idx = occupied_sites[_k]
            neighbors = nb_get_1D_neighbors(idx, L)
            ## If the site contains both predator and prey, simply pick one
            if prey_lattice[idx] and pred_lattice[idx]>0:
                _is_prey = np.random.random() < 0.5 
            elif prey_lattice[idx]:
                _is_prey = True 
            else:
                _is_prey = False 

            if _is_prey:
                ## If the site contains prey, choose a random neighboring site, and if its
                #  empty, reproduce with probability (rate) σ
                # (NOTE: empty means both eligible and no prey or predator on the site)

                # Check if neighboring sites are empty
                empty_sites = np.logical_and(
                    sites[neighbors], np.logical_and(~prey_lattice[neighbors], pred_lattice[neighbors]==0)
                )
                empty_neighbors = neighbors[empty_sites]
                _nempty = len(empty_neighbors)
                
                if _nempty > 0:
                    ## Reproduction only possible if there are empty neighboring sites
                    # Randomly sample one neighboring sites
                    _nidx = np.int64(_nn*np.random.random())
                    neighbor = neighbors[_nidx]
                    _is_empty = empty_sites[_nidx]
                    # If empty, place prey there with probability σ
                    if _is_empty and  np.random.random() < sigma :
                        # print(t, 'placing prey')
                        prey_lattice[neighbor] = True       # Place prey on prey lattice
                        occupied_sites.append(neighbor)     # Append occupied site to the list
                        M += 1
                        K += 1
                else:                   
                    ## If it is surrounded, remove it from the occupied sites
                    # (NOTE that even though the site is still occupied, we should not take
                    # it into account in our main loop as prey cannot reproduce)
                    eligible_neighbors = empty_neighbors[
                        np.logical_and(sites[empty_neighbors], ~prey_lattice[empty_neighbors])
                    ]
                    _neligible = len(eligible_neighbors)
                    if _neligible == 0 and pred_lattice[idx] == 0:
                        occupied_sites[_k], occupied_sites[-1] = occupied_sites[-1], occupied_sites[_k]
                        del occupied_sites[-1]
                        K -= 1
                        not_counted_lattice[idx] = True
            else:
                ## If the site contains a predator, check in order
                # (i)   die with mortality rate μ
                # (ii)  start a new flight
                # (iii) continue the current flight
                # (iv)  interact with pray, to possibly consume and/or reproduce

                ## Get predator ID from the predator lattice
                _pred_id = pred_lattice[idx] - 1
                ## (i) die with mortality rate μ
                if np.random.random() < mu:
                    pred_lattice[idx] = 0           # Remove predator from predator lattice
                    occupied_sites[_k], occupied_sites[-1] = occupied_sites[-1], occupied_sites[_k]
                    del occupied_sites[-1]
                    N -= 1
                    K -= 1
                else:
                    ## (ii) start a new flight
                    if curr_length[_pred_id] == 0:
                        flight_length[_pred_id] = nb_sample_powlaw_discrete(alpha)
                        didx[_pred_id] = np.random.randint(0,4)
                    
                    ## (iii) continue the current (or just started) flight
                    # Determine new 1D index using periodic boundary conditions
                    _i = ( idx // L + delta_idx_2D[didx[_pred_id]][0] ) % L
                    _j = ( idx %  L + delta_idx_2D[didx[_pred_id]][1] ) % L 
                    new_idx = _i * L + _j
                    # Ensure single occupancy by doing nothing when the site is
                    # already occupied by a predator or predators
                    if pred_lattice[new_idx] > 0:
                        curr_length[_pred_id] = 0       # Truncate current flight
                    else:
                        # Increment current path length
                        curr_length[_pred_id] += 1
                        # End current flight if current path length exceeds sampled length
                        if curr_length[_pred_id] > flight_length[_pred_id]:
                            curr_length[_pred_id] = 0 

                        ## (iv) interact with prey        
                        if prey_lattice[new_idx]:
                            _r = np.random.random()
                            ## (iv)(a) do not interact with prey with probability 1-Λ
                            if _r < 1 - Lambda_:                                
                                # Update predator lattice
                                pred_lattice[new_idx] = pred_lattice[idx]
                                pred_lattice[idx] = 0 
                                occupied_sites[_k] = new_idx
                            ## (iv)(b) reproduce onto the prey site with probability Λ*λ 
                            elif 1-Lambda_ < _r <  1-Lambda_+Lambda_*lambda_:
                                # Append new predator to appropriate lists
                                flight_length.append(0)         # Reset its flight length
                                curr_length.append(0) 
                                didx.append(0)
                                # Update predator lattice
                                pred_lattice[new_idx] = current_max_id
                                current_max_id += 1
                                N += 1
                                # Update prey lattice
                                prey_lattice[new_idx] = False   # Remove prey from that site
                                curr_length[_pred_id] = 0       # Truncate current flight
                                M -= 1
                            ## (iv)(c) consume (replace) prey with probability Λ*(1-λ)
                            else:
                                # Remove the site previously occupied by the predator
                                occupied_sites[_k], occupied_sites[-1] = occupied_sites[-1], occupied_sites[_k]
                                del occupied_sites[-1]
                                K -= 1
                                # Update predator lattice 
                                pred_lattice[new_idx] = pred_lattice[idx]
                                pred_lattice[idx] = 0
                                # Update prey lattice
                                prey_lattice[new_idx] = False   # Remove prey from that site
                                curr_length[_pred_id] = 0       # Truncate current flight
                                M -= 1
                                # ## (iv)(b) Reproduce onto the site with rate λ
                                # if np.random.random() < lambda_:
                                #     pred_lattice[new_idx] = current_max_id
                                #     flight_length.append(0)
                                #     curr_length.append(0)
                                #     didx.append(0)
                                #     current_max_id += 1
                                #     N += 1
                                # ## (iv)(c) Consume the predator and take its place with 1-λ
                                # else:
                                #     # Remove the site previously occupied by the predator
                                #     occupied_sites[_k], occupied_sites[-1] = occupied_sites[-1], occupied_sites[_k]
                                #     del occupied_sites[-1]
                                #     K -= 1
                                #     # Update the predator lattice 
                                #     pred_lattice[new_idx] = pred_lattice[idx]
                                #     pred_lattice[idx] = 0
                                
                            # If the site was previously not counted due to prey being surrounded
                            # on all sides, we need to re-include these sites and all its neighbors
                            if not_counted_lattice[new_idx]:
                                occupied_sites.append(new_idx)
                                not_counted_lattice[new_idx] = False 
                                K += 1
                                # Get the neighbors
                                _neighbors = nb_get_1D_neighbors(new_idx, L)
                                # Loop through the neighbors
                                for _idx in _neighbors:
                                    if not_counted_lattice[_idx]:
                                        occupied_sites.append(_idx)
                                        not_counted_lattice[_idx] = False 
                                        K += 1
                        else:
                            ## Displace
                            pred_lattice[new_idx] = pred_lattice[idx]
                            pred_lattice[idx] = 0 
                            occupied_sites[_k] = new_idx
    
    return prey_population, pred_population, coexistence, lattice_configuration

#################################
# Wrapper for the numba modules #
class SLLVM(object):
    """ Class for the Stochastic Lattice Lotka-Volterra Model """
    def __init__(self, seed) -> None: 
        self.Lattice = src.lattice.Lattice(seed)
        # Fix the RNG
        np.random.seed(seed)
        nb_set_seed(seed)

    def run_system(self, args):
        # Compute the prey (resource) sites on the L x L lattice
        if args.rho == 1:
            sites = np.ones((2**args.m, 2**args.m), dtype=np.int64)
        else:
            _lattice = self.Lattice.SpectralSynthesis2D(2**args.m, args.H)
            sites = self.Lattice.binary_lattice(_lattice, args.rho)
        # Initialize dictionary
        outdict = {}
        # Run 
        output = nb_SLLVM(
            args.T, args.N0, args.M0, sites, 
            args.mu, args.lambda_, args.Lambda_, args.sigma, args.alpha,
            args.nmeasures, args.visualize
        )
        # Save
        outdict['prey_population'] = output[0]
        outdict['pred_population'] = output[1]
        outdict['coexistence'] = output[2]
        if args.visualize:
            outdict['sites'] = sites 
            outdict['lattice'] = output[3]
        return outdict

    
