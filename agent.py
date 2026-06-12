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
        
        self.environment = CVRP_env("super_teste", 10000, False, device=self.device)

        self.policy = GAT_Policy(node_dim=5, edge_attr=1, hidden_dim=32)
        self.policy.to(device=self.device)

        self.target = GAT_Policy(node_dim=5, edge_attr=1, hidden_dim=32)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.to(device=self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)


    class ReplayMemory (object):
        def __init__(self, capacity=100):
            self.memory = deque([], maxlen=capacity)

        def __len__ (self):
            return len(self.memory)
        
        def push (self, args):
            self.memory.append(args[:])

        def sample (self):
            return random.choice(self.memory)
        

    def train (self):
        state = self.environment._get_graph_state()
        #trained = False
        for episode in range(self.iterations):

            losses = []
            
            replay_memory = self.ReplayMemory()
            #print(replay_memory.memory)
            #print(state.x)
            done = False
            max_steps = 3*self.environment.state.nodes


            for i in range(max_steps):
                if done == True:
                    state = self.environment.reset().to(self.device)
                    break

                experience = []
                experience.append(state)

                if random.random() < self.epsilon and self.epsilon > self.eps_threshold: # resolver mask e visited
                    action = random.choice([idx for idx, val in enumerate(self.environment.get_mask()) if val == 1])
                    self.epsilon = self.epsilon*self.decay
                else:
                    with torch.no_grad():
                        actions = self.policy(state)

                    action = self.environment.get_masked_action(actions)


                #print(action)
                state, reward, done = self.environment.step(action)
                
                experience.extend([reward, state])

                replay_memory.push(experience)

                
                if i%4 == 0:
                    #print(replay_memory.sample())
                    #print(replay_memory.memory)
                    #print(replay_memory.sample())
                    state_0, reward, state_1 = replay_memory.sample()

                    if done:
                        output = reward

                    else:
                        #print(state_1.x)
                        state_1_action_space = self.target(state_1)

                        available = [node[1] for node in state_1.x]
                        exceed_cap = [node[3] >= node[0] for node in state_1.x]
                        
                        mask = torch.tensor(available)*torch.tensor(exceed_cap)

                        state_1_action = int(torch.argmax(mask.to(self.device)*state_1_action_space))
                        print(state_1_action)
                    
                        state_1_reward = -state_1.x[state_1_action][4]

                        output = reward + self.gamma*state_1_reward
                    
                    #print(state_0.x)
                    state_0_action_space = self.policy(state_0)

                    available = [node[1] for node in state_0.x]
                    exceed_cap = [node[3] >= node[0] for node in state_0.x]
                    #print(available)
                    #print(exceed_cap)
                    mask = torch.tensor(available)*torch.tensor(exceed_cap)
                    #print(mask)
                    
                    state_0_action = int(torch.argmax(mask.to(self.device)*state_0_action_space))
                    
                    state_0_reward = -state_0.x[state_0_action][4]
                    
                    criterion = torch.nn.MSELoss().to(self.device)
                    loss = criterion(torch.tensor(state_0_reward, requires_grad=True).to(self.device), torch.tensor(output, requires_grad=True).to(self.device))
                    losses.append(loss)
                    
                    loss.requires_grad_()
                    self.optimizer.zero_grad()
                    loss.backward()
                    print(f"Episode: {episode}, Step:{i}, loss:{loss}")
            
                
            self.target.load_state_dict(self.policy.state_dict())
            self.target.to(device=self.device)