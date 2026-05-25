

class DeepQAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 5e-5
        self.epsilon = 1.0
        self.decay = 0.999
        self.eps_threshold = 0.01
        