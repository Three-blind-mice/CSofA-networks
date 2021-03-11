from base.base_generator import BaseDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


class DataGenerator(BaseDataGenerator):

    def __init__(self, config):
        super(DataGenerator, self).__init__(config)
        data = pd.read_csv(self.config.glob.path_to_csv, squeeze=True)
        data['Label'] = LabelEncoder().fit_transform(y=data['Class'])
        if len(self.config.data_loader.split.split_sizes) == 2:
            train_size, val_size = tuple(self.config.data_loader.split.split_sizes)
            random_state = self.config.data_loader.split.random_state
            train_files, val_files, train_labels, val_labels = train_test_split(
                data,
                data['Class'],
                train_size=train_size,
                test_size=val_size,
                random_state=random_state,
                stratify=data['Class']
            )
            self.train_files, self.val_files = tuple(
                map(pd.DataFrame, [train_files, val_files]))
            self.train_files['Class'] = train_labels
            self.val_files['Class'] = val_labels
            self.train_generator = self._get_generator(files=train_files, is_train=True)
            self.val_generator = self._get_generator(files=val_files, is_train=False)
        else:
            print('Error of split size')
            raise

    def get_train_data(self):
        return self.train_generator

    def get_valid_data(self):
        return self.val_generator

    def get_data(self):
        return self.get_train_data(), self.get_valid_data()

    def _get_image_generator(self, is_train=True):
        if is_train:
            return ImageDataGenerator(
                rescale=1./self.config.data_loader.augmentation.rescale,
                rotation_range=self.config.data_loader.augmentation.rotation_range,
                brightness_range=self.config.data_loader.augmentation.brightness_range,
                horizontal_flip=self.config.data_loader.augmentation.horizontal_flip,
                shear_range=self.config.data_loader.augmentation.shear_range,
                zca_epsilon=self.config.data_loader.augmentation.zca_epsilon,
                zoom_range=self.config.data_loader.augmentation.zoom_range,
                width_shift_range=self.config.data_loader.augmentation.width_shift_range,
                height_shift_range=self.config.data_loader.augmentation.height_shift_range
            )
        else:
            return ImageDataGenerator(
                rescale=1./self.config.data_loader.augmentation.rescale
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



