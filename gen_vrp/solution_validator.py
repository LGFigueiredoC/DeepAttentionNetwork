
class Solution_validator ():
    def __init__(self):
        pass

    def get_route_distance (self, instance, routes):
        def get_distance (instance, route):
            new_route = [0] + route + [0]
            #print("nr", new_route)
            #print(instance["edge_weight"])
            return sum([instance["edge_weight"][new_route[i]][new_route[i+1]] for i, j in enumerate(new_route) if i+1 < len(new_route)])
        
        return sum([get_distance(instance, route) for route in routes])