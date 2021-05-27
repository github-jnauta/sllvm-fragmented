""" Contains arguments that specify variables used in the simulation """
# Import necessary libraries
import argparse
import numpy as np 

class Args():
    """ Arguments """
    def __init__(self):
        parser = argparse.ArgumentParser("Specify specific variables")
        ## Landscape variables
        parser.add_argument(
            '--m', dest='m', type=int, default=7,
            help='level of resolution that defines the LxL lattice with L=2**m'
        )
        parser.add_argument(
            '--std', dest='std', type=float, default=1.,
            help='standard deviation of the Gaussian distribution'
        )
        parser.add_argument(
            '--H', dest='H', type=float, default=0.5, help='Hurst exponent'
        )
        parser.add_argument(
            '--rho', dest='rho', type=float, default=0.05, help='level of occupancy'
        )
        parser.add_argument(
            '--M0', dest='M0', type=int, default=-1, 
            help='initial number of prey (resources)'
        )
        ## Resource variables
        parser.add_argument(
            '--sigma', dest='sigma', type=float, default=0.2,
            help='specify the reproduction rate of the prey (resources)'
        )
        ## Forager variables
        parser.add_argument(
            '--T', dest='T', type=int, default=250, help='number of Monte-Carlo steps'
        )
        parser.add_argument(
            '--alpha', dest='alpha', type=float, default=2., 
            help='specify Levy parameter of the forager(s)'
        )
        parser.add_argument(
            '--mu', dest='mu', type=float, default=0.02,
            help='specify predator mortality rate'
        )
        parser.add_argument(
            '--lambda', dest='lambda_', type=float, default=0.2,
            help='specify predator reproduction rate'
        )
        parser.add_argument(
            '--N0', dest='N0', type=int, default=256,
            help='initial number of predators (foragers)'
        )
        ## Random number variables
        parser.add_argument(
            '--seed', dest='seed', type=int, default=42
        )
        ## Numerical variables
        parser.add_argument(
            '--reps', dest='reps', type=int, default=30,
            help='specify the number of repetitions per resource landscape'
        )
        parser.add_argument(
            '--k', dest='nmeasures', type=int, default=100,
            help='specify the number of times population size needs to be measured'
        )
        ## Boolean variables
        parser.add_argument(
            '--save', dest='save', action='store_true',
            help='if included, saves data/figure(s)'
        )
        parser.add_argument(
            '--no-save', dest='nosave', action='store_true',
            help='if included, explicitly does not save data/figure(s)'
        )
        parser.add_argument(
            '--compute', dest='compute', action='store_true', 
            help='if included, computed necessary quantities instead of loading them'
        )
        ## Directory variables 
        parser.add_argument(
            '--ddir', dest='ddir', type=str, default='data/',
            help='specify directory for output data'
        )
        
        # Parse arguments
        self.args = parser.parse_args()
