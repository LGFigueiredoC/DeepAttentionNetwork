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
        for episode in range(self.iterations):
            losses = []
            experience = []
            replay_memory = self.ReplayMemory()
            state = self.environment.reset().to(self.device)
            experience.append(self.environment.state.current_node)
            done = False
            max_steps = 3*self.environment.state.nodes


            for i in range(max_steps):
                if random.random() < self.epsilon and self.epsilon > self.eps_threshold: # resolver mask e visited
                    action = random.sample(self.environment.state.visited)
                    self.epsilon = self.epsilon*self.decay
                else:
                    with torch.no_grad():
                        actions = self.policy(state)

                    action = self.environment.get_masked_action(actions)


                state, reward, done = self.environment.step(action)
                experience.extend([reward, self.environment.state.current_node])

                replay_memory.push(experience)

                
                if i%4 == 0:
                    state, reward, destination = replay_memory.sample(1)

                    if done:
                        output = reward

                    else:
                        target_action = int(torch.argmax(self.target(destination)))
                        target_reward = self.environment.state.distance_matrix[target_action][destination]

                        output = reward + self.gamma*target_reward
                    
                    actual_action = torch.argmax(self.policy(state))
                    actual_reward = self.environment.state.distance_matrix[state][actual_action]

                    loss = torch.nn.MSELoss(output - actual_reward)
                    losses.append(loss)
                    
                    loss.requires_grad_()
                    self.optimizer.zero_grad()
                    loss.backward()
                break
            break


                

            