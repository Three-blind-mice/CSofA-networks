class BaseDataGenerator(object):
    def __init__(self, config):
        self.config = config
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def get_train_data(self):
        raise NotImplementedError

    def get_valid_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

