# sslvm-fragmented
Restricted stochastic lattice Lotka-Volterra Model in a fragmented environment generated by fractional Brownian motion. 
Within the lattice we deploy a predator-prey model wherein prey species are restricted to inhabit sites belonging to available fragmented habitat.
Predators can instead disperse freely on the lattice and do so following a Levy walk that implied optimal foraging behavior.

This code is used to replicate the results presented in the publication (see the pre-print on [Biorxiv](https://www.biorxiv.org/content/10.1101/2021.11.10.468021v2.full.pdf)):
J. Nauta, P. Simoens, Y. Khaluf, R. Martinez-Garcia, "Foraging behavior and patch size distribution jointly determine population dynamics in fragmented landscapes", Journal of the Royal Society Interface (2022, to appear)

If more details are necessary, please do not hesitate to reach out to (one of the) authors.

## Packages
The code runs using Python (version 3.7+), taking advantage of the numerical computation library [`numba`](https://numba.pydata.org/), `scipy` for pre-computation of the Hurwitz-ζ function, and `matplotlib` and `scipy` for plotting and fitting, respectively.

## Brief explanation of directory structure
The directory tree has the following shape:
```
├── .gitignore
├── README.md
├── analyze.py
├── bash
│   └── run_sllvm.sh
├── plot.py
├── run_system.py
├── src
│   ├── args.py
│   ├── compute_box_dimension.py
│   ├── depr
│   │   ├── stochastic_lotka_volterra.py
│   │   └── stochastic_lotka_volterra_restricted.py
│   ├── lattice.py
│   └── restricted_stochastic_lotka_volterra.py
└── supplementary
    ├── discrete_powlaw.py
    ├── example_lattice.py
    ├── lotka-volterra_system.py
    ├── patch_distribution.py
    ├── test.ipynb
    └── test.py
```

The important files are:

- `bash/run_sllvm.sh`: the main bash executable to be called by GNU parallel that parallelizes execution of `run_system.py` for a given list of variable arguments (i.e. different dispersal rates α, degree of fragmentation H, and/or demographic rates)
- `run_system.py`: wrapper that is used to run the main SLLVM model with the desired parameters given by command line arguments, called by `bash/run_sllvm.sh`
- `src/args.py`: description of arguments that can be supplied, including their default values
- `src/restricted_stochastic_lotka_volterra.py`: the main [`numba`](https://numba.pydata.org/) code for accelerated computing of the SLLVM

Other relevant files are:
- `plot.py`: used for plotting the figures 
- `analyze.py`: used for (parallel) analysis of the results of the simulation(s)
- `supplementary/`: contains test files and other short simulations for testing purposes

## Running the code
Specific instances with specific argument values can be run simply by using `python run_system.py`. 
Data will be stored in a `data/` directory that will be created if it does not exist.
For parallelization we use [GNU parallel](https://www.gnu.org/software/parallel/), and the syntax for calling this can be seen in `bash/run_sllvm.sh`. 
Note that we only tested this on our specific framework and specific nodes need to be supplied in a seperate file in `bash/variables.sh`. 
In this file, a variable called `nodes` needs to be defines that contains an array of the available nodes to run the code on (i.e. CPU nodes with multiple cores). 
Other names need to be supplied as well.
In short, the following minimal working example should supply you with the tools to specify your own nodes for parallelization purposes:
``` 
    ## bash/variables.sh

    # Generate array of names of available nodes
    # !note: node0 here is assumed to be the boss-node,
    #        i.e. the node on which `run_sllvm.sh` is called
    nodes=( "node0" "node1" "node2" "node3" "node4" )
    # Generate other needed constructs from the node array
    nodes_string=$(join_by , ${nodes[@]})
    noboss_nodes_string=$(join_by , ${nodes[@]:1})
    noboss_nodes=${nodes[@]:1}
```
