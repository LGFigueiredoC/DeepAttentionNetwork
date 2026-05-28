from environment import CVRP_env
from policy import GAT_Policy
import torch, copy

class DeepQAgent:
    def __init__(self, iterations=100, gamma=0.9, lr=5e-5, epsilon=1.0, decay=0.999, eps_threshold=0.01, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.iterations = iterations
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.eps_threshold = eps_threshold
        self.device = device
        
        self.environment = CVRP_env("instances", 10, True, device=self.device)
        self.policy = GAT_Policy(6, 32)
        self.target = copy.deepcopy(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)

    
    def train (self):
        max_steps = self.environment.state.nodes
        losses = []
        returns_history = []

        for episode in range(self.iterations):
            self.policy.reset_memory()
            state = self.environment.reset().to(self.device)

            
        