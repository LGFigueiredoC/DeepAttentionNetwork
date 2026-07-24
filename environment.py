import numpy as np
from gen_vrp import instance_generator, instance_loader
import torch
import random
from torch_geometric.data import Data
import os


class CVRP_env:
    def __init__(self, path, dimensions, instance_cap=100, generate=False, reset=False, has_solution=False, seed=42, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        if generate:
            os.makedirs(path, exist_ok=True)
            os.makedirs(path+"/instances", exist_ok=True)
            dimensions = dimensions
            attributes = [
                [f'{i}' for i in range(1, 4)],
                [f'{i}' for i in range(1, 4)],
                [f'{i}' for i in range(1, 8)],
                [f'{i}' for i in range(1, 7)]
            ]
            self.generator = instance_generator.Instance_generator(path+"/instances", n_instances=instance_cap, attributes=attributes, dimensions=dimensions)
            self.generator.reset_instance_dir()

        self.loader = instance_loader.Instance_loader(path, has_solution=has_solution, reset=reset)
        self.seed = seed
        self.depot = 0
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.reset()


    class State:
        def __init__(self, instance):
            print(instance)
            self.nodes = instance["dimension"]
            self.customer_nodes = self.nodes-1
            self.demands = np.array(instance["demand"])
            self.available = np.array([1 for i in range(self.nodes)])
            self.n_visited = 0
            self.total_capacity = instance["capacity"]
            self.remaining_capacity = self.total_capacity
            self.current_node = 0 #depot
            self.total_cost = 0
            self.step_count = 0
            self.time_step = 0
            self.distance_matrix = np.array(instance["edge_weight"])

    

    def __generate_edge_index (self):
        edge_index = [[i, j] for i in range(self.state.nodes) for j in range(i+1, self.state.nodes)]

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
    

    def reset (self):
        self.current_instance = self.loader.next_instance(method="symmetric")
        self.state = self.State(self.current_instance)
        self.edge_index = self.__generate_edge_index()

        return self._get_graph_state()

        
    def _get_graph_state (self):
        customer_features = [[
                self.state.demands[i+1],
                self.state.available[i+1],
                i+1 == self.state.current_node,
                self.state.remaining_capacity,
                self.state.distance_matrix[self.state.current_node][i+1] 
            ]
            for i in range(self.state.customer_nodes)
        ]
        depot_features = [self.depot, self.state.current_node != self.depot, self.state.current_node == self.depot, self.state.remaining_capacity, self.state.distance_matrix[self.state.current_node][self.depot]]

        node_features = [depot_features] + customer_features

        x = torch.tensor(node_features, dtype=torch.float, device=self.device)

        src, tgt = self.edge_index
        edge_attr = torch.tensor(self.state.distance_matrix[src.cpu(), tgt.cpu()].copy(), dtype=torch.float32, device=self.device).detach().requires_grad_(True).unsqueeze(1)

        return Data(x=x, edge_index=self.edge_index, edge_attr=edge_attr).to(self.device)
    

    def step (self, destination):
        self.state.step_count += 1

        cost = self.state.distance_matrix[self.state.current_node][destination]
        self.state.total_cost += cost
        self.state.available[destination] = 0
        self.state.n_visited += 1

        if destination == self.depot:
            self.state.remaining_capacity = self.state.total_capacity
            #self.state.available[destination] = 1
            self.state.n_visited -= 1
        else:
            self.state.available[self.depot] = 1
        
        self.state.remaining_capacity -= self.state.demands[destination]

        self.state.current_node = destination

        if self.state.current_node == self.depot:
            #print(self.state.n_visited, self.state.customer_nodes)
            if self.state.n_visited == self.state.customer_nodes:
                print(f"DONE, {self.state.step_count} steps, {self.state.n_visited} visited")
                #self.state.remaining_capacity = self.state.total_capacity
                return self._get_graph_state(), -cost, True
            
        return self._get_graph_state(), -cost, False


    def get_mask (self):
        exceed_cap = np.array([1 if (self.state.demands[node] <= self.state.remaining_capacity) else 0 for node in range(self.state.nodes)])
        mask = self.state.available * exceed_cap
        mask[self.state.current_node] = 0
        #print("av", self.state.available, "ec", exceed_cap)
        #print(self.state.current_node, self.state.remaining_capacity, mask)
        #print(len(self.state.demands))
        return mask

    def get_masked_action (self, states):
        mask = torch.from_numpy(self.get_mask()).to(self.device)
        actions = mask*states
        action_space = torch.where(actions == 0, torch.tensor(float('-inf')), actions)

        action = torch.argmax(action_space)
        #print(action)
        return action
