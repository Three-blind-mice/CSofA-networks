class BaseTrain:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def train(self):
        raise NotImplementedError