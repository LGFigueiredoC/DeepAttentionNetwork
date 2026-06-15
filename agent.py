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
        
        self.environment = CVRP_env("mini_teste", 10, False, device=self.device)

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
        replay_memory = self.ReplayMemory()
        #trained = False
        print("Número de parâmetros treináveis:", sum(p.numel() for p in self.policy.parameters() if p.requires_grad))
        for episode in range(self.iterations):

            losses = []
            
            #print(replay_memory.memory)
            #print(state.x)
            done = False
            max_steps = 3*self.environment.state.nodes


            for i in range(max_steps):
                if done == True:
                    state = self.environment.reset().to(self.device)
                    break

                experience = []
                experience.append(state.clone())

                if random.random() < self.epsilon and self.epsilon > self.eps_threshold: # resolver mask e visited
                    mask = [idx for idx, val in enumerate(self.environment.get_mask()) if val == 1]
                    #print(mask)
                    action = random.choice(mask)
                    self.epsilon = self.epsilon*self.decay
                else:
                    with torch.no_grad():
                        actions = self.policy(state).squeeze(-1)

                    #print("policy actions", actions)
                    action = self.environment.get_masked_action(actions)


                #print(action)
                state, reward, done = self.environment.step(action)
                
                experience.extend([reward, state.clone(), action])

                replay_memory.push(experience)

                
                if i%4 == 0:
                    #print(replay_memory.sample())
                    #print(replay_memory.memory)
                    #print(replay_memory.sample())
                    state_0, reward, state_1, action = replay_memory.sample()

                    if done:
                        output = torch.tensor(reward, dtype=torch.float32).to(self.device)

                    else:
                        #print(state_1.x)
                        with torch.no_grad():
                            state_1_action_space = self.target(state_1).squeeze(-1)

                        #print(state_1_action_space)
                        available = state_1.x[:, 1]
                        #print(len(available))
                        exceed_cap = state_1.x[:, 3] >= state_1.x[:, 0]
                        
                        mask = available*exceed_cap.float()
                        masked_space = state_1_action_space.masked_fill(mask == 0, -1e9)
                        
                        #print(len(masked_space))
                        state_1_action = torch.argmax(masked_space)
                        #print(state_1_action_space)
                        #print(state_1_action)
                    
                        state_1_reward = state_1_action_space[state_1_action].detach()

                        output = reward + self.gamma*state_1_reward
                    
                    #print(state_0.x)
                    state_0_action_space = self.policy(state_0).squeeze(-1)
                    #print(state_0_action_space)
                    # available = state_0.x[:, 1]
                    # exceed_cap = state_0.x[:, 3] >= state_0.x[:, 0]
                    #print(available)
                    #print(exceed_cap)
                    # mask = available*exceed_cap.float()
                    # masked_space = state_0_action_space.masked_fill(mask == 0, -1e9)
                    #print(mask)
                    
                    # state_0_action = torch.argmax(masked_space)
                    
                    state_0_reward = state_0_action_space[action]
                    
                    criterion = torch.nn.MSELoss().to(self.device)
                    loss = criterion(state_0_reward, output)
                    losses.append(loss)
                    
                    #loss.requires_grad_()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    print(f"Episode: {episode}, Step:{i}, loss:{loss}")

                    for name, param in self.policy.named_parameters():
                        if param.grad is not None:
                            print(f"Camada: {name} | Gradiente Máximo: {param.grad.abs().max().item()}")
                        else:
                            print(f"Camada: {name} | GRADIENTE É NULO!")
            
            if episode % 50 == 0:
                self.target.load_state_dict(self.policy.state_dict())
                self.target.to(device=self.device)