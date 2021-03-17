from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from base.base_train import BaseTrain
from models.model_setup import optimizers
from callbacks.cyclic_lr import CyclicLR
from plot.plot_functions import plot_history
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import numpy as np
import os
import time


class ConvModelTrainer(BaseTrain):

    def __init__(self, config, model):
        super(ConvModelTrainer, self).__init__(config, model)
        self.callbacks = []
        self._init_callbacks()

    def train(self, train_data, val_data):
        if self.config.trainer.mode.lower() == 'with_fine_tuning':
            count_steps = len(self.config.trainer.frozen_per_layers)
            frozen_per_layers = self.config.trainer.frozen_per_layers
            for p, step in zip(frozen_per_layers, range(count_steps)):
                print(f"Fine tuning step: {step}/{count_steps - 1}\n")
                start_time = time.time()
                self.model.layers[0].trainable = True
                self._freeze_base_layers(p)
                self.config.trainer.optimizer.params.learning_rate /= self.config.trainer.optimizer.learning_rate_factor
                history = self._fit(train_data, val_data, step=step)
                self._save_history(history=history, step=step+1)
                self.model.load_weights(os.path.join(self.config.callbacks.checkpoint.dir, 'best_model.hdf5'))
                self.model.save(os.path.join(self.config.callbacks.checkpoint.dir, 'model_step-{}.hdf5'.format(step)))
                print(f"Fine tuning step: {step} was completed in {((time.time()-start_time)/60):.2f} min\n")
        elif self.config.trainer.mode.lower() == 'without_fine_tuning':
            self.model.layers[0].trainable = True
            history = self._fit(train_data, val_data)
            self._save_history(history=history)
            self.model.load_weights(os.path.join(self.config.callbacks.checkpoint.dir, 'best_model.hdf5'))
            self.model.save(os.path.join(self.config.callbacks.checkpoint.dir, 'model.hdf5'))
        else:
            raise
        print('Model training completed successfully')

    def evaluate(self, data):
        scores = self.model.evaluate(data, verbose=1)
        print("Achieved an accuracy of: %.2f%%\n" % (scores[1] * 100))

    def _save_history(self, history, step=0):
        path = os.path.join(self.config.graphics.dir, 'history-{}'.format(step))
        plot_history(history).savefig(path)
        print(f'Graph of history of the loss function and accuracy was saved to {path}')

    def _freeze_base_layers(self, frozen_per_layers=1.0):
        base_layers_count = len(self.model.layers[0].trainable_variables)
        fine_tune_at = int(base_layers_count * frozen_per_layers)
        for layer in self.model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False
        trainable_layers_count = len(self.model.layers[0].trainable_variables)
        print(f'Frozen layers: {fine_tune_at} out of {base_layers_count}\n')
        print(f'Trainable layers: {trainable_layers_count} out of {base_layers_count}\n')

    def _fit(self, train_data, val_data, step=0):
        optimizer_name = self.config.trainer.optimizer.name.lower()
        optimizer_params = self.config.trainer.optimizer.params.toDict()
        optimizer = optimizers[optimizer_name](**optimizer_params)
        loss_function = self.config.trainer.loss_function
        metrics = self.config.trainer.metrics
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )
        if self.config.trainer.class_weight == 'balanced':
            y_classes = list(train_data.classes)
            class_weights = compute_class_weight('balanced', np.unique(y_classes), y_classes)
            sample_weights = compute_sample_weight('balanced', y_classes)
            class_weights_dict = dict(zip(np.unique(y_classes), class_weights))
        else:
            class_weights_dict = None
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.trainer.num_epochs[step],
            verbose=self.config.trainer.verbose,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            class_weight=class_weights_dict,
            steps_per_epoch=self.config.trainer.steps_per_epoch,
            callbacks=self.callbacks,
        )
        return history

    def _init_callbacks(self, step=0):
        if self.config.callbacks.checkpoint.exist:
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.config.callbacks.checkpoint.dir, 'best_model.hdf5'),
                    monitor=self.config.callbacks.checkpoint.monitor,
                    mode=self.config.callbacks.checkpoint.mode,
                    save_best_only=self.config.callbacks.checkpoint.save_best_only,
                    save_weights_only=self.config.callbacks.checkpoint.save_weights_only,
                    verbose=self.config.callbacks.checkpoint.verbose,
                )
            )
        if self.config.callbacks.tensor_board.exist:
            self.callbacks.append(
                TensorBoard(
                    log_dir=self.config.callbacks.tensor_board.log_dir,
                    write_graph=self.config.callbacks.tensor_board.write_graph,
                )
            )
        if self.config.callbacks.early_stopping.exist:
            self.callbacks.append(
                EarlyStopping(
                    monitor=self.config.callbacks.early_stopping.monitor,
                    patience=self.config.callbacks.early_stopping.patience,
                    restore_best_weights=self.config.callbacks.early_stopping.restore_best_weights
                )
            )
        if self.config.callbacks.cyclic_lr.exist:
            self.callbacks.append(
                CyclicLR(
                    base_lr=self.config.callbacks.cyclic_lr.base_lr,
                    max_lr=self.config.callbacks.cyclic_lr.max_lr,
                    step_size=self.config.callbacks.cyclic_lr.step_size,
                    mode=self.config.callbacks.cyclic_lr.mode,
                    gamma=self.config.callbacks.cyclic_lr.gamma
                )
            )
        if self.config.callbacks.reduce_lr_on_plateau.exist:
            self.callbacks.append(
                ReduceLROnPlateau(
                    monitor=self.config.callbacks.reduce_lr_on_plateau.monitor,
                    factor=self.config.callbacks.reduce_lr_on_plateau.factor,
                    patience=self.config.callbacks.reduce_lr_on_plateau.patience,
                    min_lr=self.config.callbacks.reduce_lr_on_plateau.min_lr
                )
            )


