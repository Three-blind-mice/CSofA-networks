from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from base.base_train import BaseTrain
from models.model_setup import optimizers
from callbacks.cyclic_lr import CyclicLR
from plot.plot_functions import plot_history
import os


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
                self._freeze_base_layers(p)
                self.config.trainer.optimizer.params.learning_rate /= self.config.trainer.optimizer.learning_rate_factor
                history = self._fit(train_data, val_data, step=step)
                print(f"Fine tuning step: {step}/{count_steps}, current val accuracy: {history.history['val_acc']}")
                plot_history(history)

        elif self.config.trainer.mode.lower() == 'without_fine_tuning':
            history = self._fit(train_data, val_data)
            plot_history(history)
        else:
            raise

    def _freeze_base_layers(self, frozen_per_layers=1.0):
        base_layers_count = len(self.model.layers[0].layers)
        fine_tune_at = int(base_layers_count * frozen_per_layers)
        for layer in self.model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False
        print('Frozen {} layers out of {} \n'.format(fine_tune_at, base_layers_count))

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
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.trainer.num_epochs[step],
            verbose=self.config.trainer.verbose,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            class_weight=self.config.trainer.class_weight,
            steps_per_epoch=self.config.trainer.steps_per_epoch,
            callbacks=self.callbacks,
        )
        return history

    def _init_callbacks(self):
        if self.config.callbacks.checkpoint.exist:
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.config.callbacks.checkpoint.dir,
                                          '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
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


