from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras import activations
from tensorflow.keras import backend
from base.base_model import BaseModel
from models.model_setup import transfer_models


class ConvModel(BaseModel):
    def __init__(self, config):
        super(ConvModel, self).__init__(config)
        self._build_model()

    def _build_model(self):
        input_shape = (*tuple(self.config.glob.image_size), self.config.glob.image_n_chanel)
        if self.config.model.transfer_model.exist:
            self.transfer_model = transfer_models[self.config.model.transfer_model.name](
                weights=self.config.model.transfer_model.weights,
                include_top=False,
                input_shape=input_shape
            )
        else:
            self.transfer_model = None
        self.model = Sequential([
            self.transfer_model,
            GlobalAveragePooling2D()
        ])
        n_layers = len(self.config.model.dense.units)
        dense_units = self.config.model.dense.units
        dense_activation = self.config.model.dense.activation
        dropout_rates = self.config.model.dropout.rates
        n_classes = self.config.glob.n_classes
        if dense_activation in 'tanh_softplus':
            dense_activation = lambda x: x * backend.tanh(activations.softplus(x))
        for units, dropout_rate, _ in zip(dense_units, dropout_rates, range(n_layers)):
            self.model.add(Dense(units, activation=dense_activation))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=dropout_rate))
        self.model.add(Dense(n_classes, activation='softmax'))

        base_layers_count = len(self.model.layers[0].trainable_variables)
        fine_tune_at = int(base_layers_count * 0.5)
        for layer in self.model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False
