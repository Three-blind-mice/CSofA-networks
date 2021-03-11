class BasePredictor(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    def predict(self):
        raise NotImplementedError
