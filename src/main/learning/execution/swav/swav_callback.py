import os
import logging
import tensorflow as tf


class SwAVCallback(tf.keras.callbacks.Callback):
    def __init__(self, features_model: tf.keras.models.Model, projections_model: tf.keras.models.Model):
        super(tf.keras.callbacks.Callback, self).__init__()

        self.working_dir = os.getenv('WORK_DIR', 'src/')
        self.logger = logging.getLogger(__name__)

        features_model_weights_file_path = os.path.join(self.working_dir,
                                                        'ckpt',
                                                        'feature-model-weights.{epoch:02d}-{loss:.2f}.hdf5')
        self.logger.debug('Features Detection Weights to be saved on: %s', features_model_weights_file_path)

        proj_model_weights_file_path = os.path.join(self.working_dir,
                                                           'ckpt',
                                                           'projections-model-weights.{epoch:02d}-{loss:.2f}.hdf5')
        self.logger.debug('Projections Weights to be saved on: %s', proj_model_weights_file_path)

        self.feature_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=features_model_weights_file_path,
                                                                         monitor='loss',
                                                                         save_best_only=True,
                                                                         save_weights_only=True,
                                                                         load_weights_on_restart=True)
        self.feature_model_callback.set_model(features_model)

        self.projections_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=proj_model_weights_file_path,
                                                                             monitor='loss',
                                                                             save_best_only=True,
                                                                             save_weights_only=True,
                                                                             load_weights_on_restart=True)
        self.projections_model_callback.set_model(projections_model)

    def on_train_begin(self, logs=None):
        try:
            self.logger.debug('on_train_begin started with logs: %s.', logs)

            self.feature_model_callback.on_train_begin(logs)
            self.projections_model_callback.on_train_begin(logs)

            self.logger.debug('on_train_begin completed.')
        except Exception as e:
            self.logger.error(e)

    def on_epoch_begin(self, epoch, logs=None):
        try:
            self.logger.debug('on_epoch_begin for epoch %s started with logs: %s.', epoch + 1, logs)

            self.feature_model_callback.on_epoch_begin(epoch=epoch, logs=logs)
            self.projections_model_callback.on_epoch_begin(epoch=epoch, logs=logs)

            self.logger.debug('on_epoch_begin completed.')
        except Exception as e:
            self.logger.error(e)

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.logger.debug('on_epoch_end started.')

            self.feature_model_callback.on_epoch_end(epoch=epoch, logs=logs)
            self.projections_model_callback.on_epoch_end(epoch=epoch, logs=logs)

            self.logger.debug('on_epoch_end completed.')
        except Exception as e:
            self.logger.error(e)
