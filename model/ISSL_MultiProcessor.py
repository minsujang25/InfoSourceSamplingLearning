"""
ISSL MultiProcessor

This script facilitates parallel execution of the InfoSourceSamplingLearning (ISSL) model 
using Python's multiprocessing module. It defines worker functions that execute individual 
model runs and collect relevant simulation outputs.

Dependencies:
    - multiprocessing
    - tqdm
    - dill
    - numpy
    - itertools
    - collections
    - warnings
"""

import multiprocessing
from tqdm.auto import tqdm
import dill
import os
import numpy as np
import copy
from itertools import product
from collections import OrderedDict
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# Set recursion limit (be cautious with this)
sys.setrecursionlimit(1000000)

# Worker function
def worker(iter_args):
    """
    Worker function to run a simulation model instance and extract key attributes.

    This function is designed to be executed in parallel processes. It initializes a model
    instance using the provided model constructor and configuration parameters, runs the simulation,
    and then extracts key outputs such as agent belief trajectories, vote results, and adversary outcomes.

    Parameters:
        iter_args containing:
            - model_i (class): The model class to instantiate.
            - model_attribute (dict): A dictionary of parameters for the model.
            - iteration (int): The iteration number (used for indexing or logging purposes).

    Returns:
        dict: A dictionary containing key attributes from the model simulation:
            - 'agent_mu_theta_list_last_step' (list): Agent parameters (mu/theta) from the final simulation step.
            - 'agent_mu_theta_list_half_step' (list): Agent parameters from the midpoint of the simulation.
            - 'comparison_rule' (str): The rule used for comparing information source credibility assessment.
            - 'epsilon' (float): The epsilon value for epsilon-greedy sampling startegy. The epsilon value identifies the probability of exploration.
            - 'iteration' (int): Iteration per model definition.
            - 'initial_theta_type' (str): Either "Flat", "Polarized", or "Consensual" used for citizens' initial theta beliefs distribution.
            - 'steps_run' (int): How many steps were run until the model reaches the equilibrium.
            - 'surveil_ability' (int): Adversarial Jammer's surveillancce ability.
            - 'num_max_citizen_neighbor' (int): The maximum number of citizen agents each citizen can communicate with. Either [0, 1, 2, 3, 4, 5].
            - 'network' (networkx object): The Network structure used in the model.
            - 'network_type' (str): Network building methods: either baseline, random, group_id_matching, or clustered.
    """
    model_i = iter_args[0]
    model_attribute = iter_args[1]
    iteration = iter_args[2]
    outcome = {}

    model = model_i(model_attribute = model_attribute)
    model.run_model(print_time=False)

    half_steps = int(model.schedule.steps/2)

    outcome['agent_mu_theta_list_last_step'] = model.agent_mu_theta_list[-1]
    outcome['agent_mu_theta_list_half_step'] = model.agent_mu_theta_list[half_steps]
    outcome['comparison_rule'] = model.comparison_rule
    outcome['epsilon'] = model.epsilon
    outcome['iteration'] = iteration
    outcome['initial_theta_type'] = model.initial_theta_type
    outcome['steps_run'] = model.schedule.steps
    outcome['surveil_ability'] = model.surveillance_ability
    outcome['num_max_citizen_neighbor'] = model.num_max_citizen_neighbor
    outcome['network'] = model.network
    outcome['network_type'] = model.mode
    return outcome

def _make_model_args_mp(model, model_attributes, iterations = 100):
    """
    Model argument building function to map the model attributes to parallel processors.
    """
    total_iterations = iterations
    all_kwargs = []

    count = len(model_attributes)
    if count:
        for params in model_attributes:
            kwargs = params.copy()
            for iter in range(iterations):
                kwargs_repeated = kwargs.copy()
                all_kwargs.append(
                    [model, kwargs_repeated, iter]
                    )

    total_iterations *= count

    return all_kwargs, total_iterations


# Multiprocessing function
def run_mp(model, model_attributes, num_process=int(multiprocessing.cpu_count() ), iterations = 100):
    """
    Run multiple instances of the model in parallel.

    Parameters:
        - model (class): The model class to be executed.
        - model_attributes (list): List of model attributes to process. Each attribute is defined as a dictionary
        - num_process (int): Number of processors to use in multiprocessing.
        - iterations (int): Number of iteration to repeat simulations.

    Returns:
        A list of dictionaries containing outputs from each model run.
    """
    run_iter_args, total_iterations = _make_model_args_mp(model_attributes = model_attributes, model=model, iterations = iterations)
    output_data_list = []
    print(f"USING {num_process} Cores...")
    with tqdm(total=total_iterations, desc="Running Models") as pbar:
        with multiprocessing.Pool(processes=num_process) as pool:
            for outcome in pool.imap_unordered(worker, run_iter_args):
                output_data_list.append(outcome)
                pbar.update(1)

    return output_data_list

