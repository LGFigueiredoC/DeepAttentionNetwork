import random
import subprocess
import os


class Instance_generator:
    def __init__(self, path, n_instances, attributes, dimensions):
        self.path = path
        self.n_instances = n_instances
        self.attributes = attributes
        self.dimensions = dimensions


    def generate_instances (self):
        os.makedirs(self.path, exist_ok=True)
        # inst_path = os.path.join(self.path, "instances")
        # os.makedirs(inst_path, exist_ok=True)

        for i in range (int(self.n_instances)):
            self.generate_one_instance(i)
                

    def generate_one_instance (self, id):
        dimension = random.choice(self.dimensions)
        depot = random.choice(self.attributes[0])
        customer_positioning = random.choice(self.attributes[1])
        demand = random.choice(self.attributes[2])
        avg_route_size = random.choice(self.attributes[3])
        subprocess.run(["python3", "gen_vrp/generator.py", f'{dimension}', depot, customer_positioning,
                        demand, avg_route_size, f'{id+1}', '42', self.path])
        

    def reset_instance_dir (self):
        for file in os.listdir(self.path):
            f_path = os.path.join(self.path,file)
            if os.path.isfile(f_path):
                os.remove(f_path)
        
        self.generate_instances()
