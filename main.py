from gen_vrp import instance_generator
from environment import CVRP_env
from agent import DeepQAgent
import time
import pandas as pd
import os

def main ():
    t0 = time.time()
    iterations = 10 #numero total de iterações no treinamento
    dimensions = [20]
    model_dir = "melhores_modelos"
    instance_dir = "all_instances"
    #instance_dir é o diretório mestre, dentro deve haver um diretório "instances" que é onde as instâncias vão ser geradas.
    #instance_cap é o número máximo de instâncias que pode ficar dentro de um diretório
    #reset = True deve ser mantido para que o treinamento continue depois que as instâncias acabem
    agent = DeepQAgent(iterations=iterations, dimensions=dimensions, instance_dir="inst_geradas", instance_cap=5, reset=False, validation=True, model_dir="melhores_modelos") 
    # training_data, time_data = agent.train()
    # training_info = pd.DataFrame(training_data)
    # training_info.index = ["loss", "step"]
    
    # saves_info = pd.DataFrame(time_data)
    # saves_info.index = ["save_time", "step", "loss"]

    # os.makedirs("training_data", exist_ok=True)
    # training_info.to_csv("training_data/training_info.csv")
    # saves_info.to_csv("training_data/saves_info.csv")
    #model_data = agent.validate(instance_dir, "melhores_modelos/modelo_20_normalizado.pth")
    # greedy_data = agent.greedy_algorithm("mini_teste")

    
    # greedy_data.sort(key=lambda x: x[0])

    #m_d = pd.DataFrame(model_data, columns=["name", "model_distance", "solution", "percentage", "model_time"])
    #m_d.set_index(m_d.columns[0])
    # g_d = pd.DataFrame(greedy_data, columns=["name", "greedy_distance", "greedy_time"], index="name")

    #print(m_d)
    # print(g_d)
    # print(g_d+m_d)
    
    print(f"Tempo total de execução: {time.time()-t0}")

    models = os.listdir(model_dir)

    for model in models:
        model_data = pd.DataFrame(columns=["name", "model_distance", "solution", "percentage", "model_time"])
        model_path = os.path.join(model_dir, model)
        print(model_path)
        for i in range(5):
            
            result = agent.validate(instance_dir, model_path=model_path)
            result.sort(key=lambda x: x[0])
            print(len(result))

            data = pd.DataFrame(result, columns=["name", "model_distance", "solution", "percentage", "model_time"])

            model_data = pd.concat([model_data, data], ignore_index=True)

        #print(model_data)
        model_data.to_csv(f"resultados/result_{os.path.splitext(model)[0]}.csv")
        #break
    #print(f"Total de Episódios: {iterations}")

if __name__ == "__main__":
    main()