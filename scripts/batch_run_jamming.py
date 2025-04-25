### Batch Run for the Baseline Model with 1 biased/precise vs 1 unbiased/precise ###

## Import required modules

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import copy
from InfoSourceSamplingLearning import Citizen, DisruptiveJammer, InfoProvider, InfoSampleModel
from ISSL_MultiProcessor import run_mp
import dill
import warnings


## Define global variables to run model ##
np.random.seed(100)

NUMNODES = N = 502

def make_snm_digraph(num_nodes=NUMNODES, num_ips = 2):
    """
    Make a social network model's DiGraph 
    """
    edges_list = []
    ips_list = list(range(0,num_ips))
    citizen_list = list(range(num_ips, num_nodes+1))
    node_list = ips_list + citizen_list
    for ip in ips_list:
        for citizen in citizen_list:
            edges_list.append((citizen, ip))
            
    DG = nx.DiGraph(name="social_network_model")
    DG.add_nodes_from(node_list)
    DG.add_edges_from(edges_list)
    return DG

g=make_snm_digraph()

mu_delta = [[0,0,0,0] for i in range(N)]
sd_delta = [[5,5,5,5] for i in range(N)]

mu_theta = [0] + [4] + list(np.random.uniform(-5,5,N-2))
sd_theta = [1] + [1] + [5 for i in range(N-2)]

type_of_agent = [InfoProvider]+ [DisruptiveJammer] + [Citizen] * (N-2)
epsilon_list = [0.05]
comparison_rule_list= ['delta_comparison','z_stat_comparison']
mu_theta_type_list=['flat','polarized','consensus']
learn_method_list = ['cautious']
surveil_ability = [2**i for i in range(9)] + [500]

mu_theta_polarized = [0] + [4] + list(np.random.normal(-3,1,size=int((N-2)/2))) + list(np.random.normal(3,1,size=int((N-2)/2)))
mu_theta_unified = [0] + [4] + list(np.random.normal(0,1,size=N-2))

mu_theta_list = [mu_theta]
mu_theta_list.append(mu_theta_polarized)
mu_theta_list.append(mu_theta_unified)

num_max_citizen_neighbor = [ i for i in range( 5+1 ) ]

network_type = ['random', 'group_id_matching']

param_dict={
    "state_of_the_world":0,
    "num_nodes": N,
    "comparison_rule": comparison_rule_list[0],
    "epsilon":epsilon_list[0],
    "credit":20,
    "mu_delta": mu_delta,
    "sd_delta": sd_delta,
    "mu_theta": mu_theta_list[0],
    "sd_theta": sd_theta,
    "seq_meaningful":True,
    "type_of_agent":type_of_agent,
    "max_steps":10000,
    "network_type": 'fully_connected', # either 'fully_connected' or 'social_network_model' or 'manually_defined'
    "mode": network_type[0], # 'incidental_learning_allowed', or  'random', or  'group_id_matching' or 'baseline'
    "network_structure": None, # default = None; should be defined when 'network_type' == 'manually_defined'
    "learn_method": learn_method_list[0], # or "cautious" or "selective"m
    'counterpart_pick_mechanism': 'equal',
    'surveil_ability': surveil_ability[0],
    'num_max_citizen_neighbor': num_max_citizen_neighbor[0]
}

# create a list of model definitions for batch runner 
param_dict_list=[]
for i in range(len(epsilon_list)):
    param_dict1 = copy.deepcopy(param_dict)
    param_dict1['epsilon'] = epsilon_list[i]
    for j in range(len(comparison_rule_list)):
        param_dict2=copy.deepcopy(param_dict1)
        param_dict2['comparison_rule'] = comparison_rule_list[j]
        for k in range(len(mu_theta_list)):
            param_dict3=copy.deepcopy(param_dict2)
            param_dict3['mu_theta'] = mu_theta_list[k]
            param_dict3['initial_theta_type'] = mu_theta_type_list[k]
            for max_num_citizen in num_max_citizen_neighbor:
                param_dict4=copy.deepcopy(param_dict3)
                param_dict4['num_max_citizen_neighbor'] = max_num_citizen
                for s_a in surveil_ability:
                    param_dict5=copy.deepcopy(param_dict4)
                    param_dict5['surveil_ability'] = s_a
                    for net_typ in network_type:
                        param_dict6=copy.deepcopy(param_dict5)
                        param_dict6['mode'] = net_typ
                        param_dict_list.append(param_dict6)



### Run Batch Runner ###
if __name__ == "__main__":
    # define batch runner
    warnings.filterwarnings("ignore")
    
    batch_results = run_mp(model = InfoSampleModel, model_attributes = param_dict_list, iterations = 100)

    # run models until it reaches to the equilibria
    with open("batch_result/batch_result(n=500)_jamming.pkl", "ab") as file:
        dill.dump(batch_results, file)