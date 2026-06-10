from environment import CVRP_env
from policy import GAT_Policy
import torch, copy
from collections import deque
import random

class DeepQAgent:
    def __init__(self, iterations=100, gamma=0.9, lr=5e-5, epsilon=1.0, decay=0.999, eps_threshold=0.01, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.iterations = iterations
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.eps_threshold = eps_threshold
        self.device = device
        
        self.environment = CVRP_env("test_instances", 10, False, device=self.device)
        self.policy = GAT_Policy(node_dim=5, edge_attr=1, hidden_dim=32)
        self.target = copy.deepcopy(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)

    class ReplayMemory (object):
        def __init__(self, capacity=100):
            self.memory = deque([], maxlen=capacity)

        def __len__ (self):
            return len(self.memory)
        
        def push (self, args):
            self.memory.append(args)

        def sample (self, batch_size):
            return random.sample(self.memory, batch_size)
        

    def train (self):
        max_steps = 3*self.environment.state.nodes
        losses = []
        replay_memory = self.ReplayMemory()
        done = False

        for episode in range(self.iterations):
            #self.policy.reset_memory()
            experience = []
            state = self.environment.reset().to(self.device)
            experience.append(self.environment.state.current_node)
            #print(state)
            for i in range(max_steps):
                #print(state)
                action = self.policy(state)
                #print(action)

                state, reward, done = self.environment.step(int(torch.argmax(action)))
                experience.extend([self.environment.state.current_node, reward])

                replay_memory.push(experience)
                print(i,state, reward, done)
                # if done:
                #     output = reward

                # else:
                #     target_action = self.target(state)
                #     output = reward + self.gamma*int(torch.argmax(target_action))

                #  loss = torch.nn.MSELoss(output - replay_memory.sample())
                break

            break


                

            