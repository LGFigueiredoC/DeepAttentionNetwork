from gen_vrp import instance_generator
from environment import CVRP_env
from agent import DeepQAgent

def main ():
    agent = DeepQAgent(iterations=10)
    agent.train()

if __name__ == "__main__":
    main()