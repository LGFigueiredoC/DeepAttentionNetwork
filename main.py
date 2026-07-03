from gen_vrp import instance_generator
from environment import CVRP_env
from agent import DeepQAgent
import time
import pandas as pd

def main ():
    t0 = time.time()
    iterations = 10 #numero total de iterações no treinamento
    #instance_dir é o diretório mestre, dentro deve haver um diretório "instances" que é onde as instâncias vão ser geradas.
    #instance_cap é o número máximo de instâncias que pode ficar dentro de um diretório
    #reset = True deve ser mantido para que o treinamento continue depois que as instâncias acabem
    agent = DeepQAgent(iterations=iterations, instance_dir="mini_teste", instance_cap=5, reset=True, model_dir="models_teste") 
    losses, episodes = agent.train()
    data = pd.DataFrame([losses, episodes])
    
    data.to_csv("training_info.csv")

    # model_data = agent.validate("mini_teste", "models_normalized/model_50000.pth")
    # greedy_data = agent.greedy_algorithm("mini_teste")

    # model_data.sort(key=lambda x: x[0])
    # greedy_data.sort(key=lambda x: x[0])

    # m_d = pd.DataFrame(model_data, columns=["name", "model_distance", "model_time"], index=map(0))
    # g_d = pd.DataFrame(greedy_data, columns=["name", "greedy_distance", "greedy_time"], index="name")

    # print(m_d)
    # print(g_d)
    # print(g_d+m_d)
    
    print(f"Tempo total de execução: {time.time()-t0}")
    #print(f"Total de Episódios: {iterations}")

if __name__ == "__main__":
    main()