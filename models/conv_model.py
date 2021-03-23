from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras import activations
from tensorflow.keras import backend
from base.base_model import BaseModel
from models.model_setup import transfer_models, regularizers, optimizers


class ConvModel(BaseModel):
    def __init__(self, config):
        super(ConvModel, self).__init__(config)
        self._build_model()

    def _build_model(self, transfer_model=None, kernel_regularizer=None, kernel_initializer=None):
        input_shape = (*tuple(self.config.glob.image_size), self.config.glob.image_n_chanel)
        if self.config.model.transfer_model.exist:
            transfer_model = transfer_models[self.config.model.transfer_model.name](
                weights=self.config.model.transfer_model.weights,
                include_top=False,
                input_shape=input_shape
            )
        self.model = Sequential([
            transfer_model,
            GlobalAveragePooling2D()
        ])
        n_layers = len(self.config.model.dense.units)
        dense_units = self.config.model.dense.units
        dense_activation = self.config.model.dense.activation
        dropout_rates = self.config.model.dropout.rates
        n_classes = self.config.glob.n_classes
        if dense_activation == 'tanh_softplus':
            dense_activation = lambda x: x * backend.tanh(activations.softplus(x))
        if self.config.model.kernel_regularizer.exist:
            kernel_regularizer_name = self.config.model.kernel_regularizer.name
            alpha = self.config.model.kernel_regularizer.alpha
            kernel_regularizer = regularizers[kernel_regularizer_name](alpha)
        if self.config.model.kernel_initializer.exist:
            kernel_initializer = self.config.model.kernel_initializer.name
        for units, dropout_rate, _ in zip(dense_units, dropout_rates, range(n_layers)):
            self.model.add(Dense(
                units,
                activation=dense_activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer
            ))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=dropout_rate))
        self.model.add(Dense(n_classes, activation='softmax'))

