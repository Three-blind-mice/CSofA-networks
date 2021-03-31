from tensorflow.keras.applications import DenseNet201, VGG19, Xception
from tensorflow.keras.optimizers import SGD, Adamax, Adam, Adadelta, Adagrad, Nadam
from efficientnet.tfkeras import EfficientNetB6, EfficientNetB7
from tensorflow.keras.regularizers import l1, l2
import numpy as np


transfer_models = {
    'dense_net_201': DenseNet201,
    'efficient_net_b6': EfficientNetB6,
    'efficient_net_b7': EfficientNetB7,
    'xception': Xception,
    'vgg_19': VGG19
}

optimizers = {
    'adam': Adam,
    'sgd': SGD,
    'adamax': Adamax,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'nadam': Nadam
}

regularizers = {
    'l1': l1,
    'l2': l2
}

# def exp_decay(epoch):
#    initial_lrate = 0.1
#    k = 0.1
#    lrate = initial_lrate * exp(-k*t)
#    return lrate