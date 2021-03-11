from base.base_predictor import BasePredictor


class Predictor(BasePredictor):

    def __init__(self, config, model):
        super(Predictor, self).__init__(config, model)
