from gen_vrp import instance_generator
from environment import CVRP_env
from agent import DeepQAgent
import time

def main ():
    t0 = time.time()
    iterations = 50000
    agent = DeepQAgent(iterations=iterations)
    agent.train()

    print(f"Tempo total de execução: {time.time()-t0}")
    print(f"Total de Episódios: {iterations}")

if __name__ == "__main__":
    main()