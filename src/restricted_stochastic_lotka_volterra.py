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
import numpy as np 
import numba
from scipy.special import zeta 
# Import modules
import src.lattice

###################
# Numba functions #
# --------------- #
# General methods
@numba.jit(nopython=True, cache=True)
def nb_set_seed(seed):
    """ Specifically call the seed from Numba code, as calling numpy.random.seed()
        from non-Numba code (or from object mode code) will seed the Numpy random 
        generator, not the Numba random generator.
    """
    np.random.seed(seed)

# Specialized methods
@numba.jit(nopython=True, cache=True)
def nb_sample_powlaw_discrete(P, xmin=1):
    """ Generate discrete sample from the truncated power law distribution 
        If there is no truncation, while other methods can sample discrete samples,
        when α->1 they often result in integer overflow due to (potentially) large
        discrete samples. Since predators with almost certainty will not finish a flight
        that has a length of several times the lattice size, we can safely truncate
        at a length L'>>L. To efficiently generate samples that obey l<L', we simply
        sample from a truncated power law.

        For this implementation, we need the Riemann zeta function, which is currently
        not automatically included in the numba library. Therefore, we need to pre-compute
        the values of the Riemann zeta function between [xmin,xmax], as we need them for
        sampling, see e.g.:
        Clauset et al., 2009, https://www.jstor.org/stable/pdf/25662336.pdf 
        Zhu et al., 2016, https://sci-hub.st/10.1111/anzs.12162 

        Parameters 
        ----------
        P: np.array(dtype=np.float64)
            The complementary cumulative distribution function of the power law
            distributed variable x, as Pr(X≥x)
    """
    # Use binary search to pinpoint the integer k for which:
    # (i)  k ≤ x < k+1
    # (ii) P(x) = 1-r, with r∈[0,1]
    # The integer k (which is the integer part of x), is then the sample
    # NOTE: NumPy's searchsorted uses binary search to pinpoint the index of where to
    #       insert a value such that the array is still sorted (i.e. for which (i) and
    #       subsequently (ii) hold). However, NumPy sorts from low to high, while the
    #       complementary cumulative distribution is given from high to low. Hence we 
    #       need to revert the order of P. Additionally, we need to give the index of 
    #       the non-reversed array, which is given by len(P)-idx-1. Finally, we add xmin.
    r = np.random.random()
    idx = np.searchsorted(P[::-1], 1-r)
    sample = xmin + len(P) - idx - 1
    return sample

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

## Main function
# @profile
@numba.jit(nopython=True, cache=True)
def nb_SLLVM(
        T, N0, M0, sites, reduced_sites, mu, lambda_, Lambda_, sigma, alpha, P, P_reduced,
        sites_patch_dict, reduced_sites_patch_dict,
        nmeasures, bins, visualize, xmin=1, reduce=False
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
        P : np.array(xmax-xmin, dtype=np.float64)
            The complementary cumulative distribution function of the flight lengths
            which are power law distributed variables
        sites_patch_dict : dict(types.int64 : np.array([], dtype=np.int64))
            A list of arrays that contains indices of the seperate patches of the lattice
        reduced_sites_patch_dict : dict(types.int64 : np.array([], dtype=np.int64))
            A list of arrays that contains indices of the seperate patches of the reduced lattice
        bins : np.array(nbins, dtype=np.int64)
            The (log-spaced) bins for determining the distribution over flight lengths
        nmeasures : np.int64
            Number of times populations are measures at equally spaced intervals
        visualize : np.bool_
            Boolean flag that includes lattice configuration over time in return
        xmin : np.int64
            Minimum predator displacement. Default set to 1 as the lattice size.
    """
    # Specify some variables
    L, _ = sites.shape              # Size of LxL lattice
    rho = np.sum(sites) / L**2      # Habitat density
    dmeas = T // (nmeasures -1)     # Δt at which measurements need to be collected
    treduce = T // 2                # Time at which habitat will be reduced
    _nn = 4                         # Number of nearest neighbors
    # Adapt some variables as they should take on a specific value if -1 is provided
    mu = 1 / L if mu == -1 else mu                          # Death rate     
    N0 = np.int64(min(0.2,rho)*L**2) if N0 == -1 else N0    # Initial number of predators
    # Ensure the complementary CDF of the inverse power law is handles properly
    P = P_reduced if not reduce else P 

    ## Initialize constants
    delta_idx_2D = [[0,1], [0,-1], [1,0], [-1,0]]

    ## Convert the eligible sites to 1D array
    sites = sites.flatten()
    reduced_sites = reduced_sites.flatten()
    ## Initialize the 1-dimensional lattices
    #  We need three lattices:
    #  * one lattice of integer type that contains the IDs of the predators
    #  * one lattice of boolean type that contains prey
    #  * one lattice of boolean type that contains sites not counted (see above)
    pred_lattice = np.zeros(L*L, dtype=np.int64)
    prey_lattice = np.zeros(L*L, dtype=np.bool_)
    not_counted_lattice = np.zeros(L*L, dtype=np.bool_)

    ## Distribute prey on eligible sites
    prey_sites = np.flatnonzero(sites==1)
    M0 = N0 if M0 == -1 else M0 
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
    Lambda_lst = [1. for i in gen]      # Detection probability of predator
    current_max_id = N0 + 1
    ## Allocate other lists
    empty_labels = [np.int64(i) for i in range(0)]

    ## Initialize
    N = N0          # Number of predators       
    M = M0          # Number of prey
    K = N0 + M0     # Number of occupied sites

    ## Allocate arrays for storing measures
    prey_population = np.zeros(nmeasures+1, dtype=np.int64)
    pred_population = np.zeros(nmeasures+1, dtype=np.int64)
    habitat_efficiency = np.zeros(nmeasures+1, dtype=np.int64)
    predators_on_habitat = np.zeros(nmeasures+1, dtype=np.int64)
    isolated_patches = np.zeros(nmeasures+1, dtype=np.int64)
    coexistence = 1
    ## Allocate logaritmically spaced bins for flight length distribution
    flight_lengths = np.zeros(len(bins)-1, dtype=np.int32)

    # Store initial values
    prey_population[0] = M0 
    pred_population[0] = N0 
    habitat_efficiency[0] = np.count_nonzero((pred_lattice+prey_lattice)[sites])
    predators_on_habitat[0] = np.count_nonzero(pred_lattice[sites])
    sites_patchlabels = [label for label in sites_patch_dict]
    for label in sites_patch_dict:
        patch_indices = sites_patch_dict[label]
        if label not in empty_labels and not np.any(prey_lattice[patch_indices]):
            isolated_patches[0] += len(patch_indices)
            empty_labels.append(label)

    # Allocate and initialice lattice configuration if visualize flag is given
    lattice_configuration = np.zeros((L*L, nmeasures+1), dtype=np.int16)
    if visualize:
        lattice_configuration[sites>0,0] = 1
        lattice_configuration[prey_lattice,0] = -1 
        lattice_configuration[pred_lattice>0,0] = 2

    ##############################################
    ## Run the stochastic Lotka-Volterra system ##
    for t in range(1,T+1):
        ## Store desired variables every dmeas timesteps
        if t % dmeas == 0:
            imeas = t // dmeas
            # Store prey and predator population
            prey_population[imeas] = M 
            pred_population[imeas] = N
            # Store habitat efficiency
            habitat_efficiency[imeas] = np.count_nonzero((pred_lattice+prey_lattice)[sites])
            predators_on_habitat[imeas] = np.count_nonzero(pred_lattice[sites])
            # Store the number of isolated (dead) patches
            for i, label in enumerate(sites_patchlabels):
                patch_indices = sites_patch_dict[label]
                if label not in empty_labels and not np.any(prey_lattice[patch_indices]):
                    isolated_patches[imeas] += len(patch_indices)
                    # Do not count in future computations, as once a patch is depleted
                    # it will (and should) never become rehabitated.
                    empty_labels.append(label)
            # Store the lattice configuration
            if visualize:
                lattice_configuration[sites>0,imeas] = 1
                lattice_configuration[prey_lattice,imeas] = -1 
                lattice_configuration[pred_lattice>0,imeas] = 2

        ## Stop the simulation if:
        # prey goes extinct, as predators will also go extinct
        if M == 0:
            coexistence = 0
            break
        # predators go extinct, as prey will fully occupy all available sites
        if N == 0:
            coexistence = 0
            prey_population[imeas:] = np.sum(sites)
            habitat_efficiency[imeas:] = np.sum(sites)
            break 

        ## Decrease the number of habitable sites halfway through the simulation
        if reduce == True and t == treduce:
            # Loop through all prey and kill them if the site is now inhabitable
            for i in np.flatnonzero(prey_lattice):
                if reduced_sites[i] == 0:
                    prey_lattice[i] = False
                    M -= 1
            # Ensure that the new eligible sites are the reduced ones
            sites = reduced_sites 
            sites_patch_dict = reduced_sites_patch_dict
            sites_patchlabels = [label for label in sites_patch_dict]
            empty_labels = [np.int64(i) for i in range(0)]
            # Recompute list of occupied sites
            prey_idxs = np.flatnonzero(prey_lattice)
            pred_idxs = np.flatnonzero(pred_lattice)
            occupied_sites = [idx for idx in prey_idxs] + [idx for idx in pred_idxs]  
            # Disregard all not counted sites          
            not_counted_lattice = np.zeros(L*L, dtype=np.bool_)
            # Reset number of occupied sites
            K = N + M 
            # Reconfigure the complementary CDF, as predators will now be sampling with α
            P = P_reduced

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
                    if _is_empty and  np.random.random() < sigma:
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
                        # Sample new flight length
                        flight_length[_pred_id] = nb_sample_powlaw_discrete(P, xmin)
                        didx[_pred_id] = np.random.randint(0,4)
                        # Update detection probability
                        if Lambda_ == -1:
                            Lambda_lst[_pred_id] = 1 / flight_length[_pred_id]
                    
                    ## (iii) continue the current (or just started) flight
                    # Determine new 1D index using periodic boundary conditions
                    _i = ( idx // L + delta_idx_2D[didx[_pred_id]][0] ) % L
                    _j = ( idx %  L + delta_idx_2D[didx[_pred_id]][1] ) % L 
                    new_idx = _i * L + _j
                    # Ensure single occupancy by doing nothing when the site is
                    # already occupied by a predator or predators
                    if pred_lattice[new_idx] > 0:
                        # Count flight length for the distribution 
                        if 0 < curr_length[_pred_id] < np.max(bins):
                            bin = np.searchsorted(bins, curr_length[_pred_id])
                            flight_lengths[bin] += 1
                        curr_length[_pred_id] = 0       # Truncate current flight                        
                    else:
                        # Increment current path length
                        curr_length[_pred_id] += 1
                        ## (iv) interact with prey        
                        if prey_lattice[new_idx]:
                            _r = np.random.random()
                            ## (iv)(a) do not interact with prey with probability 1-Λ
                            if _r < 1 - Lambda_lst[_pred_id]:
                                # Update predator lattice
                                pred_lattice[new_idx] = pred_lattice[idx]
                                pred_lattice[idx] = 0 
                                occupied_sites[_k] = new_idx
                            ## (iv)(b) reproduce onto the prey site with probability Λ*λ 
                            elif 1-Lambda_lst[_pred_id] < _r <  1-Lambda_lst[_pred_id]+Lambda_lst[_pred_id]*lambda_:
                                # Append new predator to appropriate lists
                                flight_length.append(0)         # Reset its flight length
                                curr_length.append(0) 
                                didx.append(0)
                                Lambda_lst.append(1.)
                                # Update predator lattice
                                pred_lattice[new_idx] = current_max_id
                                current_max_id += 1
                                N += 1
                                # Update prey lattice
                                prey_lattice[new_idx] = False   # Remove prey from that site                                
                                # Count flight length for the distribution 
                                if curr_length[_pred_id] < np.max(bins):
                                    bin = np.searchsorted(bins, curr_length[_pred_id])
                                    flight_lengths[bin] += 1
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
                                # Count flight length for the distribution 
                                if curr_length[_pred_id] < np.max(bins):
                                    bin = np.searchsorted(bins, curr_length[_pred_id])
                                    flight_lengths[bin] += 1
                                curr_length[_pred_id] = 0       # Truncate current flight
                                M -= 1
                                
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

                        # End current flight if current path length exceeds sampled length
                        if curr_length[_pred_id] >= flight_length[_pred_id]:
                            # Count flight length for the distribution 
                            if curr_length[_pred_id] < np.max(bins):
                                bin = np.searchsorted(bins, curr_length[_pred_id])
                                flight_lengths[bin] += 1
                            curr_length[_pred_id] = 0
    
    return prey_population, pred_population, coexistence, flight_lengths, habitat_efficiency, predators_on_habitat, isolated_patches, empty_labels, lattice_configuration

#################################
# Wrapper for the numba modules #
class SLLVM(object):
    """ Class for the Stochastic Lattice Lotka-Volterra Model """
    def __init__(self, seed) -> None: 
        self.Lattice = src.lattice.Lattice(seed)
        # Fix the RNG
        np.random.seed(seed)
        nb_set_seed(seed)

    def run_system(self, args, xmin=1, xmax=None):
        # Compute the prey (resource) sites on the L x L lattice
        L = 2**args.m 
        _lattice = self.Lattice.SpectralSynthesis2D(L, args.H)
        sites = self.Lattice.binary_lattice(_lattice, args.rho)
        reduced_sites = self.Lattice.binary_lattice(_lattice, args.rho/5)
        # Gather indices of the seperate patches of the habitable patches
        sites_patch_dict, num_patches = self.Lattice.label(sites)
        reduced_sites_patch_dict, num_reduced_patches = self.Lattice.label(reduced_sites)
        # Compute maximum flight length 
        xmax = 200*L if not xmax else xmax 
        xmax_measure = 2*L if not xmax else xmax 
        # Pre-compute the bins for distribution over flight lenghts
        bins = np.logspace(np.log10(xmin), np.log10(xmax_measure), num=args.nbins, dtype=np.int64)
        bins = np.unique(bins)
        # Pre-compute the Riemann zeta function for sampling of discrete power law variables
        flightlengths = np.arange(xmin, xmax)        
        P = np.zeros(len(flightlengths))
        P[0] = 1.
        if args.alpha == -1 or args.alpha == np.inf:
            P_reduced = P 
        else:
            norm = zeta(args.alpha, xmin) - zeta(args.alpha, xmax)
            P_reduced = (zeta(args.alpha, flightlengths) - zeta(args.alpha, xmax))/norm         
        # Initialize dictionary
        outdict = {}
        # Run 
        output = nb_SLLVM(
            args.T, args.N0, args.M0, sites, reduced_sites,
            args.mu, args.lambda_, args.Lambda_, args.sigma, args.alpha, P, P_reduced,
            sites_patch_dict, reduced_sites_patch_dict,
            args.nmeasures, bins, args.visualize
        )
        # Analyze depleted patches
        empty_labels = output[7]
        patchbins = np.logspace(0, np.log10(args.rho*L**2+1), num=args.nbins, dtype=np.int64)
        patchhist = np.zeros(len(patchbins), dtype=np.int32)
        patchsizes = np.zeros(len(patchbins))        
        for label in empty_labels:
            bin = np.searchsorted(bins, len(sites_patch_dict[label]))
            patchhist[bin] += 1
        for label, site_indices in sites_patch_dict.items():
            bin = np.searchsorted(bins, len(site_indices))
            patchsizes[bin] += 1
        prob_patch_depletion = np.ma.divide(patchhist, patchsizes).filled(0)
        # Save
        outdict['prey_population'] = output[0]
        outdict['pred_population'] = output[1]
        L = 2**args.m 
        # outdict['coexistence'] = output[2]
        outdict['flight_lengths'] = output[3]
        outdict['habitat_efficiency'] = output[4]
        outdict['predators_on_habitat'] = output[5]
        outdict['isolated_patches'] = output[6]
        outdict['num_patches'] = num_patches
        outdict['num_reduced_patches'] = num_reduced_patches
        outdict['prob_patch_depletion'] = prob_patch_depletion
        if args.visualize:
            outdict['lattice'] = output[-1]
        return outdict

    
