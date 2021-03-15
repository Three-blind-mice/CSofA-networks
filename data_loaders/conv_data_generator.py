from base.base_generator import BaseDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os


class DataGenerator(BaseDataGenerator):

    def __init__(self, config):
        super(DataGenerator, self).__init__(config)
        train_data = pd.read_csv(self.config.glob.path_to_train, squeeze=True)
        images_id_list = os.listdir(self.config.glob.path_to_images)
        train_data = train_data[train_data['Id'].isin(images_id_list)]
        test_data = pd.read_csv(self.config.glob.path_to_test, squeeze=True)
        test_data = test_data[test_data['Id'].isin(images_id_list)]
        self.config.glob.n_classes = train_data['Class'].nunique()
        if len(self.config.data_loader.split.split_sizes) == 2:
            train_size, val_size = tuple(self.config.data_loader.split.split_sizes)
            random_state = self.config.data_loader.split.random_state
            train_files, val_files, train_labels, val_labels = train_test_split(
                train_data,
                train_data['Class'],
                train_size=train_size,
                test_size=val_size,
                random_state=random_state,
                stratify=train_data['Class']
            )
            self.train_files, self.val_files = tuple(
                map(pd.DataFrame, [train_files, val_files]))
            self.train_files['Class'] = train_labels
            self.val_files['Class'] = val_labels
            self.train_generator = self._get_generator(files=train_files, is_train=True)
            self.val_generator = self._get_generator(files=val_files, is_train=False)
            self.test_generator = self._get_generator(files=test_data, is_train=False)
        else:
            print('Error of split size')
            raise

    def get_train_data(self):
        return self.train_generator

    def get_valid_data(self):
        return self.val_generator

    def get_test_data(self):
        return self.test_generator

    def get_data(self):
        return self.get_train_data(), self.get_valid_data(), self.get_test_data()

    def _get_image_generator(self, is_train=True):
        if is_train:
            return ImageDataGenerator(
                **self.config.data_loader.augmentation.toDict()
            )
        else:
            return ImageDataGenerator(
                rescale=self.config.data_loader.augmentation.rescale
            )

    def _get_generator(self, files, is_train=True):
        if is_train:
            shuffle = self.config.data_loader.shuffle
        else:
            shuffle = False
        return self._get_image_generator(is_train=True).flow_from_dataframe(
            dataframe=files,
            x_col="Id",
            y_col="Class",
            shuffle=shuffle,
            save_format='jpg',
            directory=self.config.glob.path_to_images,
            target_size=tuple(self.config.glob.image_size),
            batch_size=self.config.data_loader.generator.batch_size,
            class_mode=self.config.data_loader.generator.class_mode,
            seed=self.config.data_loader.generator.random_state,
        )



