from base.base_generator import BaseDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator(BaseDataGenerator):

    def __init__(self, config):
        super(DataGenerator, self).__init__(config)
        self.train_generator = self._get_generator(path=self.config.glob.path_to_train, is_test=False)
        self.val_generator = self._get_generator(path=self.config.glob.path_to_valid, is_test=False)
        self.test_generator = self._get_generator(path=self.config.glob.path_to_test, is_test=True)

    def get_train_data(self):
        return self.train_generator

    def get_valid_data(self):
        return self.val_generator

    def get_test_data(self):
        return self.test_generator

    def get_data(self):
        return self.get_train_data(), self.get_valid_data(), self.get_test_data()

    def _get_image_generator(self, is_test=True):
        if is_test:
            return ImageDataGenerator(
                rescale=self.config.data_loader.augmentation.rescale
            )
        else:
            return ImageDataGenerator(
                **self.config.data_loader.augmentation.toDict()
            )

    def _get_generator(self, path, is_test=True):
        if is_test:
            shuffle = False
            batch_size = 1
        else:
            shuffle = self.config.data_loader.shuffle
            batch_size = self.config.data_loader.generator.batch_size
        return self._get_image_generator(is_test=True).flow_from_directory(
            directory=path,
            shuffle=shuffle,
            save_format='jpg',
            target_size=tuple(self.config.glob.image_size),
            batch_size=batch_size,
            class_mode=self.config.data_loader.generator.class_mode,
            seed=self.config.data_loader.generator.random_state,
        )



