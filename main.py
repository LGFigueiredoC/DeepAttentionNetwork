from gen_vrp import instance_generator
from environment import CVRP_env
from agent import DeepQAgent
import time
import pandas as pd

def main ():
    t0 = time.time()
    iterations = 50000
    agent = DeepQAgent(iterations=iterations)
    # losses, episodes = agent.train()
    model_data = agent.validate("mini_teste", "models_normalized/model_50000.pth")
    greedy_data = agent.greedy_algorithm("mini_teste")

    model_data.sort(key=lambda x: x[0])
    greedy_data.sort(key=lambda x: x[0])

    m_d = pd.DataFrame(model_data, columns=["name", "model_distance", "model_time"], index="name")
    g_d = pd.DataFrame(greedy_data, columns=["name", "greedy_distance", "greedy_time"], index="name")

    print(m_d)
    print(g_d)
    print(g_d+m_d)
    
    print(f"Tempo total de execução: {time.time()-t0}")
    #print(f"Total de Episódios: {iterations}")

if __name__ == "__main__":
    main()