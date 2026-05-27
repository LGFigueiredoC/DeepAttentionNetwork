import random
import subprocess


class Instance_generator:
    def __init__(self, path, n_instances):
        self.path = path
        self.n_instances = n_instances

    def generate_instances (self, dimensions, attributes):
        for dim in dimensions:
            for i in range (dimensions/len(dimensions)):
                demand = random.choice(attributes[0])
                depot = random.choice(attributes[1])
                customer_positioning = random.choice(attributes[2])
                avg_route_size = random.choice(attributes[4])
                subprocess.run(["python3", "generator.py", f'{dim}', depot, customer_positioning,
                                demand, avg_route_size, f'{i+1}', '42', self.path])
        