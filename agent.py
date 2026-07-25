from environment import CVRP_env
from policy import GAT_Policy
import torch, copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
from gen_vrp import solution_validator
import pandas as pd
import time
import os


class DeepQAgent:
    def __init__(self, instance_dir, instance_cap, model_dir, dimensions, validation=False, reset=False, iterations=100, gamma=0.9, lr=5e-5, epsilon=1.0, decay=0.999, eps_threshold=0.01, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.iterations = iterations
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.eps_threshold = eps_threshold
        self.device = device
        self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        self.instance_cap = instance_cap

        if not validation:
            self.environment = CVRP_env(instance_dir, dimensions=dimensions, instance_cap=self.instance_cap, generate=False, reset=reset, device=self.device)
        hidden = 128

        self.policy = GAT_Policy(node_dim=5, edge_attr=1, hidden_dim=hidden)
        self.policy.to(device=self.device)

        self.target = GAT_Policy(node_dim=5, edge_attr=1, hidden_dim=hidden)
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

        def sample (self, k):
            return random.sample(self.memory, k=k)
        

    def train (self):
        start_time = time.time()
        state = self.environment._get_graph_state()
        replay_memory = self.ReplayMemory()
        total_steps = 0
        print("Número de parâmetros treináveis:", sum(p.numel() for p in self.policy.parameters() if p.requires_grad))
        save_times = []
        time_step = []
        time_loss = []
        losses = []
        step_memory = []
        best_loss = np.inf

        for episode in range(self.iterations+1):
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
                    action = random.choice(mask)
                    self.epsilon = self.epsilon*self.decay
                else:
                    with torch.no_grad():
                        actions = self.policy(state).squeeze(-1)

                    action = self.environment.get_masked_action(actions)

                state, reward, done = self.environment.step(action)
                
                experience.extend([reward, state.clone(), action, done])

                replay_memory.push(experience)

                total_steps += 1

                if i%4 == 0 and len(replay_memory) >= 32:
                    batches = replay_memory.sample(32)
                    predictions = []
                    targets = []

                    for state_0, reward, state_1, action, replay_done in batches:
                        if replay_done:
                            output = torch.tensor(reward, dtype=torch.float32).to(self.device)

                        else:
                            with torch.no_grad():
                                state_1_action_space = self.target(state_1).squeeze(-1)

                            available = state_1.x[:, 1]
                            exceed_cap = state_1.x[:, 3] >= state_1.x[:, 0]
                            
                            mask = available*exceed_cap.float()
                            masked_space = state_1_action_space.masked_fill(mask == 0, -1e9)
                            
                            state_1_action = torch.max(masked_space)

                            output = reward + self.gamma*state_1_action
                        
                        state_0_action_space = self.policy(state_0).squeeze(-1)
                        
                        state_0_reward = state_0_action_space[action]
                        predictions.append(state_0_reward)
                        targets.append(output)

                    pred_tensor = torch.stack(predictions)
                    targ_tensor = torch.stack(targets)

                    
                    criterion = torch.nn.MSELoss().to(self.device)
                    loss = criterion(pred_tensor, targ_tensor)
                    losses.append(loss.item())
                    step_memory.append(total_steps)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    print(f"Episode: {episode}, Step:{i}, loss:{loss}")

            if episode % self.instance_cap == 0:
                self.environment.generator.reset_instance_dir()
                self.environment.loader.make_instances()
            
            if total_steps % 10000 == 0:
                for name, param in self.policy.named_parameters():
                        if param.grad is not None:
                            print(f"Camada: {name} | Gradiente Máximo: {param.grad.abs().max().item()}")
                        else:
                            print(f"Camada: {name} | GRADIENTE É NULO!")

                self.target.load_state_dict(self.policy.state_dict())
                self.target.to(device=self.device)

            if episode > 1:
                if len(losses) > 0 and losses[len(losses)-1] < best_loss:
                    torch.save(self.policy.state_dict(), f"{self.model_dir}/model_{int(episode/5000)*5000}.pth")
                    partial_time = time.time()
                    save_times.append(partial_time-start_time)
                    time_step.append(total_steps)
                    time_loss.append(losses[len(losses)-1])

                    best_loss = losses[len(losses)-1]

            if episode > 1 and episode % 5000 == 0:
                torch.save(self.policy.state_dict(), f"{self.model_dir}/model_{episode}.pth")
                partial_time = time.time()
                save_times.append(partial_time-start_time)
                time_step.append(total_steps)
                time_loss.append(losses[len(losses)-1])

                best_loss = np.inf
                plt.plot(losses)
                plt.title(f"loss{episode}")
                plt.savefig(f"plots/loss_{episode}.png")
                plt.close()

        return [losses, step_memory], [save_times, time_step, time_loss]


    def validate (self, val_set, model_path):
        self.environment = CVRP_env(val_set, 10, generate=False, reset=True, has_solution=True, device=self.device)
        self.policy.load_state_dict(torch.load(model_path, weights_only=True))
        self.policy.eval()
        data = []
        #print(self.environment.loader.instances)
        num_instances = len(self.environment.loader.instances)
        for instance in range(num_instances):
            #print(instance)
            t1 = time.time()
            state = self.environment._get_graph_state()

            done = False
            max_steps = 3*self.environment.state.nodes

            route = []
            path = []
            path.append(0)
            action = None
            for i in range(max_steps):
                if action == 0:
                    route.append(path.copy())
                    del path[:]
                    path.append(0)

                if done == True:
                    break


                with torch.no_grad():
                    actions = self.policy(state).squeeze(-1)

                action = self.environment.get_masked_action(actions)

                state, reward, done = self.environment.step(action)
                path.append(int(action.item()))

            #print(route)
            instance_path, inst = self.environment.loader.get_current_instance()
            #print(instance_path, inst)
            solution = self.environment.loader.get_current_solution()
            solver = solution_validator.Solution_validator()

            name = instance_path.split("/")[2]
            total_cost = solver.get_route_distance(inst, route)
            total_time = time.time()-t1
            data.append([name, total_cost, solution["cost"], round(total_cost*100/solution["cost"], 2), total_time])
            print(f"Instance: {instance} out of {num_instances}, Exec. time: {total_time}")
            #print("model", "name", name, "distance:", total_cost, "solution", solution["cost"], "percentage", round(total_cost*100/solution["cost"], 2))
            _ = self.environment.reset().to(self.device)

        return data
    
    def greedy_algorithm (self, val_set):
        self.environment = CVRP_env(val_set, 10, False, device=self.device)
        data = []

        for instance in range (len(self.environment.loader.instances)):
            t1 = time.time()
            state = self.environment._get_graph_state()

            done = False
            max_steps = 3*self.environment.state.nodes

            route = []
            path = []
            path.append(0)
            action = None
            for i in range(max_steps):
                if action == 0:
                    route.append(path.copy())
                    del path[:]
                    path.append(0)

                if done == True:
                    _ = self.environment.reset().to(self.device)
                    break

                mask = torch.from_numpy(self.environment.get_mask()).to(self.device)
                #mask = torch.tensor([np.inf for idx, val in enumerate(self.environment.get_mask()) if val == 1])
                #print(mask)
                #print(distance)
                distance = state.x[:, 4]
                actions = mask*distance
                action_space = torch.where(actions == 0, torch.tensor(float('inf')), actions)

                action = torch.argmin(action_space)

                state, reward, done = self.environment.step(action)
                path.append(int(action.item()))

            #print(route)
            instance_path, inst = self.environment.loader.get_current_instance()
            solver = solution_validator.Solution_validator()
            name = instance_path.split("/")[2]

            data.append([name, solver.get_route_distance(inst, route), time.time()-t1])
            print("greedy", "name", name, "distance:", solver.get_route_distance(inst, route))

        return data