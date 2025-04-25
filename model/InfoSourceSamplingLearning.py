"""Create a model that allows beliefs to propagate
over a user-defined directed graph. Nodes can be
different types of agents.
Author: MJ 
v 0.0.0.1

Note: This is built under Mesa 2.4 
"""

import math
import numpy as np
import networkx as nx
from scipy.spatial import distance
from scipy.optimize import minimize
from copy import deepcopy
from sklearn.cluster import KMeans

from mesa import Agent, Model
from mesa.time import RandomActivation, SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

import sys
import warnings

# Increase recursion limit and suppress warnings to avoid unnecessary console output.
sys.setrecursionlimit(10**6)
warnings.filterwarnings("ignore")

# CONSTANTS
PERIODS = 10              # Default number of simulation periods
NUM_NODES = 100           # Default number of nodes in the network
TYPES = ["citizen", "infoprovider"]  # Agent types
AGENT_TYPE_LIST = ['Citizen', 'InfoProvider']
CHOICE_OPTION = [0, 1]    # Options for learning (0: learn theta, 1: learn delta)
MIN_STD = 1e-5            # Minimum allowable standard deviation for message sampling

# ----------------------
# Helper Functions
# ----------------------
def make_fully_connected_digraph(num_nodes=NUM_NODES):
    """
    Create a fully connected directed graph.

    Parameters:
        num_nodes (int): Number of nodes in the graph.

    Returns:
        nx.DiGraph: A fully connected directed graph.
    """
    edges = [(i, j) for i in range(num_nodes)
                   for j in range(num_nodes) if i != j]
    DG = nx.DiGraph(name="fully_connected")
    DG.add_edges_from(edges)
    return DG
    
def make_snm_digraph(num_nodes=NUM_NODES, num_ips=2):
    """
    Create a directed graph for the social network model (SNM).

    Parameters:
        num_nodes (int): Total number of nodes in the network.
        num_ips (int): Number of information providers (IPs).

    Returns:
        nx.DiGraph: A directed graph for the social network model.
    """
    ips_list = list(range(num_ips))
    citizen_list = list(range(num_ips, num_nodes + 1))
    node_list = ips_list + citizen_list
    edges = [(citizen, ip) for ip in ips_list for citizen in citizen_list]
    DG = nx.DiGraph(name="social_network_model")
    DG.add_nodes_from(node_list)
    DG.add_edges_from(edges)
    return DG

def random_state_assignment(min_val=-1, max_val=1):
    """
    Assign a random state value uniformly between min_val and max_val.

    Parameters:
        min_val (float): Minimum value.
        max_val (float): Maximum value.

    Returns:
        float: A random state value.
    """
    return np.random.uniform(min_val, max_val)

# ----------------------
# Model Class
# ----------------------
class InfoSampleModel(Model):
    """
    Model for simulating information sampling and learning in a network of agents.

    Attributes:
        state_of_the_world (float): The true state of the world that agents aim to learn.
        num_nodes (int): Number of nodes in the network.
        comparison_rule (str): Rule for comparing information sources.
        epsilon (float): Learning / optimal-arm exploitation parameter.
        credit (int): Resources available for information requests.
        initial_theta_type (str): Initial type for theta values.
        network_type (str): Type of network ('fully_connected', 'social_network_model', or 'manually_defined').
        learn_method (str): Learning method ('naive', 'cautious', or 'selective').
        counterpart_pick_mechanism (str): Mechanism for selecting information counterparts.
        mode (str): Operational mode (e.g., 'baseline', 'random', 'group_id_matching', 'clustered').
        surveillance_ability (int): Ability of disruptive jammer to surveil citizens.
        num_max_citizen_neighbor (int): Maximum number of citizen neighbors to consider.
    """
    def __init__(self, num_nodes=NUM_NODES, network=make_fully_connected_digraph(NUM_NODES), 
        mode = 'incidental_learning_allowed', comparison_rule='delta_comparison', 
        epsilon=0.1, credit=20, network_type = 'fully_connected', 
        network_structure=None, learn_method='naive', counterpart_pick_mechanism = 'equal', 
        num_max_citizen_neighbor = 0, max_steps = 1000,
        model_attribute=None,
        state_of_the_world = random_state_assignment()):
        super().__init__()
        self.model_attribute = model_attribute

        # Set model parameters based on provided attributes or default values.
        if self.model_attribute is not None:
            self.state_of_the_world = model_attribute['state_of_the_world']
            self.num_nodes = model_attribute['num_nodes']
            self.comparison_rule = model_attribute['comparison_rule']
            self.epsilon = round(model_attribute['epsilon'], 2)
            self.credit = model_attribute['credit']
            self.initial_theta_type = model_attribute['initial_theta_type']
            self.network_type = model_attribute['network_type']
            self.learn_method = model_attribute['learn_method']  # 'naive', 'cautious', or 'selective'
            self.counterpart_pick_mechanism = model_attribute['counterpart_pick_mechanism']
            self.mode = model_attribute['mode']  # baseline, random, group_id, or clustered
            self.surveillance_ability = model_attribute['surveil_ability']
            self.num_max_citizen_neighbor = model_attribute['num_max_citizen_neighbor']
            self.max_steps = model_attribute['max_steps']
        else:
            self.state_of_the_world = state_of_the_world
            self.num_nodes = num_nodes
            self.comparison_rule = comparison_rule
            self.epsilon = epsilon
            self.credit = credit
            self.initial_theta_type = 'random'
            self.network_type = network_type
            self.learn_method = learn_method
            self.mode = mode
            self.surveillance_ability = 7
            self.num_max_citizen_neighbor = num_max_citizen_neighbor
            self.max_steps = max_steps

        # Initialize network based on type
        if self.network_type == 'fully_connected':
            self.network = make_fully_connected_digraph(num_nodes=self.num_nodes)
        elif self.network_type == 'social_network_model':
            self.network = make_snm_digraph(num_nodes=self.num_nodes)
        elif self.network_type == 'manually_defined':
            self.network = model_attribute['network_structure']
 
        # Set up grid and scheduler.
        self.grid = NetworkGrid(self.network)
        self.schedule = SimultaneousActivation(self)
        self.time = 0

        # Data tracking attributes.
        self.agent_mu_theta_list = []
        self.agent_num_request_list = []
        self.avg_agent_mu_theta_diff = 10.0
        self.avg_agent_mu_theta_diff_rate = 10.0
        self.avg_agent_sd_theta = 10.0

        # Create agents
        if self.model_attribute is not None and self.model_attribute['seq_meaningful'] is True:
            for i,node in enumerate(sorted(self.network.nodes())):
                #### mu_delta & sd_delta should be in the form of [mu_delta_0, mu_delta_1] & [sd_delta_0, sd_delta_1] ###
                its_alive=self.model_attribute['type_of_agent'][i](i,node,self, mu_delta=self.model_attribute['mu_delta'][i], sd_delta=self.model_attribute['sd_delta'][i], mu_theta=self.model_attribute['mu_theta'][i], sd_theta=self.model_attribute['sd_theta'][i])
                self.schedule.add(its_alive)
                    # Add the agent to the node, this adds .pos attr to agent
                self.grid.place_agent(its_alive, node)
            for i,node in enumerate(sorted(self.network.nodes())):
                self.network.nodes[node]["agent"].append(self.schedule.agents[i])

        elif self.model_attribute is not None and self.model_attribute['seq_meaningful'] is not True:
            for i,node in enumerate(self.network.nodes()):
                its_alive=self.model_attribute['type_of_agent'][i](i,node,self, mu_delta=self.model_attribute['mu_delta'][i], sd_delta=self.model_attribute['sd_delta'][i], mu_theta=self.model_attribute['mu_theta'][i], sd_theta=self.model_attribute['sd_theta'][i])
                self.schedule.add(its_alive)
                    # Add the agent to the node, this adds .pos attr to agent
                self.grid.place_agent(its_alive, node)

        else:
            for i,node in enumerate(self.network.nodes()):
                its_alive = self.types_and_proportions[temp_type_list[i]][1](i,node,self, mu_delta=np.random.uniform(-5,5), sd_delta=np.random.randint(1,10), mu_theta=np.random.uniform(-5,5), sd_theta=np.random.randint(1,10))
                self.schedule.add(its_alive)
                    # Add the agent to the node, this adds .pos attr to agent
                self.grid.place_agent(its_alive, node)

        self.running = True


    def get_agent_mu_theta(self):
        """
        Retrieve the current mu_theta beliefs from citizen agents.

        Returns:
            list: Latest mu_theta beliefs from agents.
        """
        if self.learn_method == 'naive':
            mu_theta_list = [a.mu_theta_beliefs[-1] for a in self.schedule.agents if a.type_of_agent == "citizen"]
        else:
            mu_theta_list = [a.mu_theta_beliefs[-1] for a in self.schedule.agents if a.type_of_agent == "citizen" and a.theta_or_delta == 0]
        return mu_theta_list

    def get_agent_mu_theta_delta(self):
        """
        Compute the mean absolute change in mu_theta beliefs for citizen agents.

        Returns:
            float: Mean absolute difference between the last two mu_theta beliefs.
        """
        mu_theta_delta = [ abs(a.mu_theta_beliefs[-2] - a.mu_theta_beliefs[-1]) for a in self.schedule.agents if a.type_of_agent == "citizen" and a.theta_or_delta == 0]
        return np.mean(mu_theta_delta)

    def get_agent_mu_theta_delta_rate(self):
        """
        Compute the mean relative change rate in mu_theta beliefs for citizen agents.

        Returns:
            float: Mean relative change rate.
        """
        if self.learn_method == 'naive':
            mu_theta_delta_rate = [ abs((a.mu_theta_beliefs[-2] - a.mu_theta_beliefs[-1])/a.mu_theta_beliefs[-2]) for a in self.schedule.agents if a.type_of_agent == "citizen" and a.mu_theta_beliefs[-2] != 0.0]

        else:
            mu_theta_delta_rate = [ abs((a.mu_theta_beliefs[-2] - a.mu_theta_beliefs[-1])/a.mu_theta_beliefs[-2]) for a in self.schedule.agents if a.type_of_agent == "citizen" and a.theta_or_delta == 0 and a.mu_theta_beliefs[-2] != 0.0]
            
        return np.mean(mu_theta_delta_rate)

    def get_agent_sd_theta(self):
        """
        Retrieve the current sd_theta beliefs from citizen agents.

        Returns:
            list: Latest sd_theta beliefs.
        """
        sd_theta_list = [a.sd_theta_beliefs[-1] for a in self.schedule.agents if a.type_of_agent == "citizen"]
        return sd_theta_list

    def get_agent_avg_sd_theta(self):
        """
        Compute the average standard deviation (sd_theta) among citizen agents.

        Returns:
            float: Mean sd_theta value.
        """
        sd_theta_list = self.get_agent_sd_theta()
        return np.mean(sd_theta_list)

    def get_agent_num_request(self):
        """
        Retrieve the number of requests made by citizen agents.

        Returns:
            list: Request counts from agents.
        """
        if self.learn_method == 'naive':
            num_request_list = [a.num_request for a in self.schedule.agents if a.type_of_agent == "citizen" and a.info_source[0].type_of_agent == 'infoprovider']
        else:
            num_request_list = [a.num_request for a in self.schedule.agents if a.type_of_agent == "citizen" and a.theta_or_delta == 0 and a.info_source[0].type_of_agent == 'infoprovider']
        
        return num_request_list

    def update_network(self):
        """
        Build a new directed graph representing the current counterpart relationships.
        Each citizen agent creates a directed edge from itself to every agent in its info_source.
        The function then stores the edge list (as returned by networkx) in the model attribute.
        """
        updated_net = nx.DiGraph()
        # Add all nodes (i.e. all agents).
        for agent in self.schedule.agents:
            updated_net.add_node(agent.unique_id)
        # Add edges from citizen to each counterpart in its info_source.
        for agent in self.schedule.agents:
            if agent.type_of_agent == "citizen" and hasattr(agent, "info_source"):
                for counterpart in agent.info_source:
                    updated_net.add_edge(agent.unique_id, counterpart.unique_id)
        # Save the network structure as a model attribute.
        self.network = updated_net

    def step(self):
        """
        Execute one simulation step:
          - Update time-dependent probabilities for exploration / exploitation.
          - Activate all agents.
          - Record summary statistics.
        """
        if self.schedule.steps == 0:
            for citizen in [a for a in self.schedule.agents if a.type_of_agent == "citizen"]:
                citizen.info_source = citizen.pick_counterpart()
            # Record the updated network structure based on current counterpart selections.
            self.update_network()


        self.p = 1/(self.time+1)
        self.p_pair = [1-self.p, self.p]
        self.schedule.step()

        mu_theta_list=self.get_agent_mu_theta()
        self.agent_mu_theta_list.append(mu_theta_list)
        if self.schedule.steps > 2:
            num_request_list=self.get_agent_num_request()
            #self.avg_agent_mu_theta_diff = self.get_agent_mu_theta_delta()
            self.avg_agent_mu_theta_diff_rate = self.get_agent_mu_theta_delta_rate()
            #self.avg_agent_sd_theta = self.get_agent_avg_sd_theta()
            #self.agent_num_request_list.append(num_request_list)
        
    def run_model(self, print_time=True):
        while self.running:
            if print_time is True:
                print("Running step :{}".format(self.time))
            
            self.step()
            if (self.avg_agent_mu_theta_diff_rate<10**-3 or self.schedule.steps > self.max_steps) :
                self.running=False
            self.time += 1

# ----------------------
# Agent Classes
# ----------------------
class InfoAgents(Agent):
    """
    Base class for agents in the model.

    Attributes:
        mu_theta (float): Belief about the state of the world (theta).
        sd_theta (float): Standard deviation of theta belief.
        mu_delta (list): Belief about delta (bais) belief.
        sd_delta (list): Standard deviation of delta (bias) belief.
        type_of_agent (str): Identifier for the agent type.
        pos: Position in the grid.
        epsilon (float): Learning parameter.
        group_id (int): Group identifier based on mu_theta.
    """
    def __init__(self, unique_id, pos, model, mu_delta, sd_delta, mu_theta, sd_theta, type_of_agent='genericagent'):
        super().__init__(unique_id, model)
        self.mu_theta = mu_theta
        self.sd_theta = sd_theta
        self.mu_delta = mu_delta
        self.sd_delta = sd_delta
        self.type_of_agent = type_of_agent

        self.pos = pos
        self.epsilon = self.model.epsilon

        if mu_theta <=0:
            self.group_id = -1
        else: self.group_id = 1

    def mu_out(self, n_req, requester):
        """
        Generate outgoing messages based on the agent's mu_theta belief.

        Parameters:
            n_req (int): Number of messages to generate.
            requester: The agent requesting the messages.

        Returns:
            list: Generated message values.
        """
        if n_req > 0:
            return list(np.random.normal(self.mu_theta, self.sd_theta, n_req))
        return []

class DisruptiveJammer(InfoAgents):
    """
    Agent that disrupts useful information learning by jamming messages.

    Attributes:
        citizen_intel (dict): Information on citizen agents (e.g., clusters).
        surveillance_ability (int): Ability to surveil citizens.
        msg_param_at_t_per_cluster (dict): Message parameters per cluster at each time step.
        expected_cluster_theta_beliefs (dict): Expected cluster theta beliefs.
    """
    def __init__(self, unique_id, pos, model, mu_delta, sd_delta, mu_theta, sd_theta, type_of_agent="disruptivejammer", surveillance_ability=5):
        super().__init__(unique_id, pos, model, mu_delta, sd_delta, mu_theta, sd_theta, type_of_agent)
        self.citizen_intel = {}
        self.surveillance_ability = self.model.model_attribute['surveil_ability']
        self.msg_param_at_t_per_cluster = {}
        self.expected_cluster_theta_beliefs = {}

    def surveil_citizen(self):
        """
        Cluster citizen agents using KMeans to monitor their theta beliefs and assign cluster memberships.
        """
        citizens = [a for a in self.model.schedule.agents if a.type_of_agent == 'citizen']
        mu_beliefs = [a.mu_theta_beliefs[0] for a in citizens]
        citizen_ids = [a.unique_id for a in citizens]
        kmeans = KMeans(n_clusters=self.surveillance_ability)
        data = np.array(mu_beliefs).reshape(-1, 1)
        kmeans.fit(data)
        labels = kmeans.predict(data)
        centroids = kmeans.cluster_centers_

        membership = {cid: label for cid, label in zip(citizen_ids, labels)}
        clusters = {}
        for label in np.unique(labels):
            clusters[label] = [a for a in citizens if membership[a.unique_id] == label]

        self.citizen_intel = {'centroids': centroids, 'membership': membership}
        self.citizen_per_cl = clusters

    def calculate_v(self, prior_std, alpha):
        """
        Calculate the v value used in updating expected cluster beliefs.

        Parameters:
            prior_std (float): Prior standard deviation.
            alpha (float): Learning parameter.

        Returns:
            float: Calculated v value.
        """
        return prior_std**2 / (prior_std**2 + 2 * alpha - 1)

    def expected_cluster_mean(self, prior_mu, prior_std, alpha, message_mean):
        """
        Compute the expected mean of cluster theta beliefs for the next time step.

        Parameters:
            prior_mu (float): Prior cluster mean.
            prior_std (float): Prior cluster standard deviation.
            alpha (float): Learning parameter.
            message_mean (float): Mean of the jamming message.

        Returns:
            float: Expected cluster mean.
        """
        v = self.calculate_v(prior_std, alpha)
        return (1 - v) * prior_mu + v * alpha * message_mean

    def expected_cluster_std(self, prior_std):
        """
        Compute the expected standard deviation of cluster theta beliefs for the next time step.

        Parameters:
            prior_std (float): Prior cluster standard deviation.

        Returns:
            float: Expected cluster standard deviation.
        """
        return prior_std**2 / (prior_std**2 + 1)

    def init_message_param(self):
        """
        Initialize message parameters and expected cluster theta beliefs for the current and next time steps.
        """
        now = self.model.time
        next_time = now + 1
        centroids = self.citizen_intel['centroids']
        clusters = self.citizen_per_cl

        initial_msg_param = {}
        initial_expected_beliefs = {}
        for cluster, citizens in clusters.items():
            initial_mu = [citizen.mu_theta_beliefs[0] for citizen in citizens]
            std_estimate = np.std(initial_mu)
            initial_msg_param[cluster] = {"avg": centroids[cluster], "std": 1}
            initial_expected_beliefs[cluster] = {"avg": centroids[cluster], "std": std_estimate}

        self.msg_param_at_t_per_cluster[now] = initial_msg_param
        self.msg_param_at_t_per_cluster[next_time] = initial_msg_param
        self.expected_cluster_theta_beliefs[now] = initial_expected_beliefs
        self.expected_cluster_theta_beliefs[next_time] = initial_expected_beliefs

    def tune_message_param(self):
        """
        Update message parameters and expected cluster theta beliefs for the next time step based on current beliefs.
        """
        now = self.model.time
        next_time = now + 1
        prior_beliefs = self.expected_cluster_theta_beliefs[now]
        current_msg_param = self.msg_param_at_t_per_cluster[now]

        posterior_beliefs = {}
        jamming_msg_param = {}
        for cluster, params in current_msg_param.items():
            prior_mu = prior_beliefs[cluster]['avg']
            prior_std = prior_beliefs[cluster]['std']
            alpha = 0.95
            v = self.calculate_v(prior_std, alpha)
            theta = self.model.state_of_the_world
            # Avoid division by zero in computing msg_mean
            if abs(v * alpha - 1) < 1e-8:
                msg_mean = prior_mu
            else:
                msg_mean = (v * alpha * theta - prior_mu - v * alpha * (1 - v) * prior_mu) / (v * alpha - 1)
            expected_avg = self.expected_cluster_mean(prior_mu, prior_std, alpha, msg_mean)
            expected_std = self.expected_cluster_std(prior_std)
            posterior_beliefs[cluster] = {"avg": expected_avg, "std": expected_std}
            jamming_msg_param[cluster] = {'avg': msg_mean, 'std': 1}

        self.msg_param_at_t_per_cluster[next_time] = jamming_msg_param
        self.expected_cluster_theta_beliefs[next_time] = posterior_beliefs

    def mu_out(self, n_req, requester):
        """
        Generate outgoing jamming messages for a requester based on the current message parameters of the requester's cluster.

        Parameters:
            n_req (int): Number of messages to generate.
            requester: The requesting agent.

        Returns:
            list: Generated message values.
        """
        req_id = requester.unique_id
        req_cluster = self.citizen_intel['membership'][req_id]
        mu_theta = self.msg_param_at_t_per_cluster[self.model.time][req_cluster]['avg']
        sd_theta = 1
        if n_req > 0:
            return list(np.random.normal(mu_theta, sd_theta, n_req))
        return []

    def step(self):
        """
        Execute one simulation step for the disruptive jammer.
          - Every 5 steps, re-surveil citizens and reinitialize message parameters.
          - Otherwise, tune message parameters for the next step.
        """
        if self.model.time % 5 == 0:
            self.surveil_citizen()
            self.init_message_param()
        else:
            self.tune_message_param()



class InfoProvider(InfoAgents):
    """
    Information provider agent that supplies information on state of the world based on their bias (defined by delta values).

    * note that they never update their beliefs about the state of the world.
    """
    def __init__(self, unique_id, pos, model, mu_delta, sd_delta, mu_theta, sd_theta, type_of_agent="infoprovider"):
        super().__init__(unique_id, pos, model, mu_delta, sd_delta, mu_theta, sd_theta, type_of_agent)
        self.delta = self.model.state_of_the_world - self.mu_theta

class Citizen(InfoAgents):
    """
    Citizen agent that updates its beliefs (mu_theta and sd_theta) based on received messages.

    Attributes:
        mu_theta_beliefs (list): History of mu_theta beliefs.
        sd_theta_beliefs (list): History of sd_theta beliefs.
        mu_delta_beliefs (list): History of mu_delta beliefs.
        sd_delta_beliefs (list): History of sd_delta beliefs.
        optimal_arm_id_history (list): History of chosen information sources.
        theta_or_delta_history (list): History indicating whether the agent learned theta or delta.
        num_request_history (list): History of request counts.
        theta_or_delta (float): Indicator for learning type (theta or delta).
    """
    def __init__(self, unique_id, pos, model, mu_delta, sd_delta, mu_theta, sd_theta, type_of_agent="citizen"):
        super().__init__(unique_id, pos, model, mu_delta, sd_delta, mu_theta, sd_theta, type_of_agent)
        self.mu_theta_beliefs = [mu_theta]
        self.sd_theta_beliefs = [sd_theta]
        # Extend delta beliefs by appending copies of the first element
        self.mu_delta_beliefs = [mu_delta + [mu_delta[0]] * self.model.num_max_citizen_neighbor]
        self.sd_delta_beliefs = [sd_delta + [sd_delta[0]] * self.model.num_max_citizen_neighbor]
        self.optimal_arm_id_history = [np.nan]
        self.theta_or_delta_history = [np.nan]
        self.num_request_history = [np.nan]
        self.theta_or_delta = np.nan
        

    def get_neighbor_list(self):
        """
        Retrieve a list of neighboring agents (excluding self) from the grid.

        Returns:
            list: Neighboring agents.
        """

        #neighbor_list = [neighbor for neighbor in self.model.grid.get_neighbors(self.pos, include_center=False)]
        #return neighbor_list 
        return set(list(self.model.grid.get_neighbors(self.pos, include_center=False)))

    def learn_theta_or_delta(self):
        """
        Randomly decide whether to learn theta (0) or delta (1) based on model probabilities.

        Returns:
            int: 0 (learn theta) or 1 (learn delta).
        """
        return np.random.choice(CHOICE_OPTION, p=self.model.p_pair)

    def ip_or_citizen(self):
        """
        Determine whether to choose an information provider or citizen as a counterpart.

        Returns:
            str: 'citizen' or 'infoprovider' based on the counterpart selection mechanism.
        """
        if self.model.counterpart_pick_mechanism == 'equal':
            p = [2/4, 2/4]
        elif self.model.counterpart_pick_mechanism == 'citizen_more':
            p = [3/4, 1/4]
        elif self.model.counterpart_pick_mechanism == 'ip_more':
            p = [1/4, 3/4]
        else: p = [1/2, 1/2]
        
        return np.random.choice(TYPES, p=p)

    def pick_counterpart(self):
        """
        Select counterpart agents (neighbors or additional citizens) based on the model's mode.

        Returns:
            list: Selected counterpart agents.
        """
        nmcn = self.model.num_max_citizen_neighbor
        num_citizen_neighbor = np.random.randint(nmcn+1) # to make sure the upper cap is the maximum number of citizen neighbors defined under model attribute.

        all_neighbors = self.get_neighbor_list()
        
        citizen_list = [ citizen for citizen in all_neighbors if citizen.type_of_agent == 'citizen' ]
        elite_info_sources = ['infoprovider', 'disruptivejammer']

        baseline_neighbors = [ infoprovider for infoprovider in all_neighbors if infoprovider.type_of_agent in elite_info_sources ] 

        if self.model.mode == 'baseline':
            return baseline_neighbors

        if self.model.mode == 'random':
            selected = np.random.choice(citizen_list, num_citizen_neighbor, replace=False)
            return baseline_neighbors + list(selected)

        elif self.model.mode == 'group_id_matching':
            in_group = [citizen for citizen in citizen_list if citizen.group_id == self.group_id]
            out_group = [citizen for citizen in citizen_list if citizen.group_id != self.group_id]
            
            if num_citizen_neighbor == 0:
                return baseline_neighbors
            
            choices = np.random.choice(['in', 'out'], size=num_citizen_neighbor, p=[0.9, 0.1])
            n_in = np.count_nonzero(choices == 'in')
            n_out = num_citizen_neighbor - n_in
            selected_in = list(np.random.choice(in_group, n_in, replace=False)) if n_in > 0 else []
            selected_out = list(np.random.choice(out_group, n_out, replace=False)) if n_out > 0 else []
            
            return baseline_neighbors + selected_in + selected_out

        elif self.model.mode == 'clustered':
            np.random.shuffle(citizen_list)
            selected = []
            for candidate in citizen_list:
                if len(selected) >= num_citizen_neighbor:
                    break
                dist = abs(self.mu_theta - candidate.mu_theta)
                power = 0.5 if candidate.group_id == self.group_id else 1
                p_connect = math.exp(-dist * power)
                if np.random.binomial(1, p_connect):
                    selected.append(candidate)
            
            return baseline_neighbors + selected
        
        return baseline_neighbors


    def sample_messages(self):
        """
        Sample messages from selected information sources based on the learning method.
        For the cautious method, request probabilities are assigned (possibly via a probability distribution).
        """
        learn_method = self.model.learn_method
        sources = self.info_source

        if learn_method == 'cautious':
            if self.theta_or_delta == 1:
                self.num_request = [round(self.model.credit/len(sources))] * len(sources)
                msgs = [sources[i].mu_out(self.num_request[i], self) for i in range(len(self.num_request))]

            else: 
                credibility_ranked_sources = self.credibility_ranked_sources
                num_sources = len(sources)

                if num_sources == 2:
                    p = [ 1-self.model.epsilon,  self.model.epsilon]
                else:
                    p = [ 1-self.model.epsilon ]  # Start with 0.95 for the first source
                    for i in range(num_sources - 2):
                        p.append((1 - sum(p)) * (1-self.model.epsilon))  # Assign 95% of the remaining probability
                    p.append(1 - sum(p))  # Ensure the last value makes the sum exactly 1

                sample_choice = list(np.random.choice(credibility_ranked_sources, p = p, size = self.model.credit))

                self.num_request = [ sample_choice.count(source) for source in credibility_ranked_sources ]

                #sample messages from the sources in credible order and probability ordered in credibility levels.        
                msgs = [credibility_ranked_sources[i].mu_out(self.num_request[i], self) for i in range(len(self.num_request))]

        self.sampled_msgs = msgs


    def learn_delta(self):
        """
        Update the agent's delta beliefs using Bayesian updating based on sampled messages.
        This is used only if self.theta_or_delta == 1 (i.e. when the agent is learning only about the credibility) 
        """
        prior_mu_theta = self.mu_theta_beliefs[-1]
        prior_sd_theta = self.sd_theta_beliefs[-1]
        
        msgs = self.sampled_msgs
        sources = self.info_source

        prior_mu_delta_dict = {source: self.mu_delta_beliefs[-1][sources.index(source)]  for source in sources}
        prior_sd_delta_dict = {source: self.sd_delta_beliefs[-1][sources.index(source)]  for source in sources}

        posterior_mu_delta_list = []
        posterior_sd_delta_list = []
        for source in sources:
            prior_mu_delta = prior_mu_delta_dict[source]
            prior_sd_delta = prior_sd_delta_dict[source]

            posterior_mu_delta = (prior_mu_delta * (prior_sd_theta**2 + 1**2) - (np.mean(msgs[sources.index(source)]) - prior_mu_theta) * prior_sd_delta**2)/(prior_sd_theta**2 + prior_sd_delta**2 + 1**2)
            posterior_mu_delta_list.append(posterior_mu_delta)

            posterior_sd_delta = (prior_sd_delta * 1**2 + prior_sd_delta**2 * prior_sd_theta**2)/(prior_sd_theta**2 + prior_sd_delta**2 + 1**2)
            posterior_sd_delta_list.append(posterior_sd_delta)

        self.mu_delta_beliefs.append(posterior_mu_delta_list)
        self.mu_delta = {source: posterior_mu_delta_list[sources.index(source)] for source in sources}
        self.sd_delta_beliefs.append(posterior_sd_delta_list)
        self.sd_delta = {source: posterior_sd_delta_list[sources.index(source)] for source in sources}

    def bayesian_update_mu_theta(self, msgs):
        """
        Perform Bayesian update of mu_theta based on sampled messages.

        Parameters:
            msgs (list): List of message values.

        Returns:
            float: Updated mu_theta.
        """
        prior_mu_theta = self.mu_theta_beliefs[-1]
        prior_sd_theta = self.sd_theta_beliefs[-1]
        mean_msgs = np.mean(msgs)
        MIN_STD = 1e-5
        std_msgs = max(np.std(msgs), MIN_STD)

        posterior_mu_theta = prior_mu_theta + (mean_msgs - prior_mu_theta)*prior_sd_theta**2/(prior_sd_theta**2 + std_msgs**2)
        return posterior_mu_theta

    def bayesian_update_sd_theta(self, msgs):
        """
        Perform Bayesian update of sd_theta based on sampled messages.

        Parameters:
            msgs (list): List of message values.

        Returns:
            float: Updated sd_theta.
        """
        prior_sd_theta = self.sd_theta_beliefs[-1]
        mean_msgs = np.mean(msgs)
        MIN_STD = 1e-5
        std_msgs = max(np.std(msgs), MIN_STD)

        posterior_sd_theta = (prior_sd_theta**2 * std_msgs**2) / (prior_sd_theta**2 + std_msgs**2)

        #### THIS LINE SHOULD BE FIXED LATER ####
        #if posterior_sd_theta > 10**5:
            #posterior_sd_theta = 10**5

        return posterior_sd_theta

    def calculate_z_stat(self):
        """
        Calculate z-statistics for each information source to assess credibility based on the difference between message means and posterior updates.
        """
        prior_sd_theta = self.sd_theta_beliefs[-1]
        
        msgs = self.sampled_msgs
        sources = self.info_source

        z_dict = {}
        for i, source in enumerate(sources):
            msg = msgs[i]
            posterior = self.bayesian_update_mu_theta(msg)
            MIN_STD = 1e-5
            sd_msg = max(np.std(msg), MIN_STD)

            z_dict[source] = abs((np.mean(msg) - posterior)/math.sqrt(1+(prior_sd_theta**2 / sd_msg**2)))

        self.z_statistics = z_dict

    def decide_optimal_arm(self):
        """
        Determine the optimal information source (arm) based on the credibility score for exploitation.
        Uses delta comparison or z-statistics as defined by the model.
        """
        if self.model.comparison_rule == "delta_comparison":
            self.learn_delta()
            credibility_dict = self.mu_delta
            for key, value in credibility_dict.items():
                # convert delta values into absolute values
                credibility_dict[key] = abs(value)
        else:
            self.calculate_z_stat()
            credibility_dict = self.z_statistics

        # Convert dictionary to a list of (key, value) tuples
        items = list(credibility_dict.items())

        # Shuffle the items to randomize order for equal values
        np.random.shuffle(items)

        # Sort the shuffled items by credibility score value in ascending order
        sorted_items = sorted(items, key=lambda x: x[1], reverse=False)

        self.credibility_ranked_sources = [item[0] for item in sorted_items]
        #credibility_ranked_sources = {}
        #for agent, cred_score in sorted_items:
            #credibility_ranked_sources[agent] = cred_score

        #self.credibility_ranked_sources_dict = credibility_ranked_sources
        #self.credibility_ranked_sources = list(credibility_ranked_sources.keys())

    def clear_messages(self):
        """
        Clear stored sampled messages.
        """
        self.sampled_msgs = []

    def step(self):
        """
        Execute a simulation step for the citizen agent:
          - At time 0, initialize neighbors and information sources.
          - In subsequent steps, update counterparts, decide on learning type, sample messages,
            update beliefs, and record history.
        """
            #if self.model.mode == 'incidental_learning_allowed':
                #self.info_source = self.pick_counterpart()
            #else: pass
        self.theta_or_delta = self.learn_theta_or_delta()
        self.sample_messages()
        msgs=self.sampled_msgs

        if self.theta_or_delta == 1:
            self.decide_optimal_arm()
            mu_theta = self.mu_theta_beliefs[-1]
            sd_theta = self.sd_theta_beliefs[-1]
        else: 
            mu_theta = self.bayesian_update_mu_theta( [m for msg in msgs for m in msg] )
            sd_theta = self.bayesian_update_sd_theta( [m for msg in msgs for m in msg] )
                
        self.mu_theta_beliefs.append(mu_theta)
        self.sd_theta_beliefs.append(sd_theta)
        self.mu_theta = mu_theta
        self.sd_theta = sd_theta
        #self.theta_or_delta_history.append(self.theta_or_delta)
        #self.optimal_arm_id_history.append(self.credibility_ranked_sources_dict)
        self.num_request_history.append(self.num_request)
        self.clear_messages()


# ----------------------
# Main Execution
# ----------------------
if __name__=="__main__":
    my_info_sampling_model = InfoSampleModel()
    #my_info_model.datacollector.get_agent_vars_dataframe()

