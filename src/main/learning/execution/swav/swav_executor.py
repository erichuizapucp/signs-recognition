import numpy as np
import tensorflow as tf

from learning.execution.base_model_executor import BaseModelExecutor
from learning.execution.swav.swav_callback import SwAVCallback
from learning.common.model_utility import SWAV


class SwAVExecutor(BaseModelExecutor):
    def __init__(self):
        super().__init__()

        # 224x224, 224x224, 96x96, 96x96, 96x96
        self.num_crops = [2, 3]

        # swapped assigment is intended only for 224x224 crops
        self.crops_for_assign = [0, 1]
        self.crop_sizes = [224, 224, 96, 96, 96]
        self.temperature = 0.1

        # To be assigned on the train_model method
        self.optimizer = None
        self.feature_backbone_model = None
        self.prototype_projection_model = None

        self.loss = self._get_loss()

        self.training_logger.info('SwAV initialized with: num_crops: %s, crops_for_assign: %s, '
                                  'temperature: %s', self.num_crops, self.crops_for_assign, self.temperature)

    def train_model(self, models, dataset, no_epochs, no_steps_per_epoch=None, **kwargs):
        def train(train_step_fn, optimizer, callback):
            try:
                self.feature_backbone_model = models[0]
                self.prototype_projection_model = models[1]
                self.optimizer = optimizer

                step_wise_loss = []
                epoch_wise_loss = []

                self.training_logger.info('SwAV Training started with: no_epochs: %s', no_epochs)

                callback.on_train_begin()
                for epoch in range(no_epochs):
                    callback.on_epoch_begin(epoch=epoch)
                    w = self.prototype_projection_model.get_layer('prototype').get_weights()
                    w = tf.transpose(w)
                    w = tf.math.l2_normalize(w, axis=1)
                    self.prototype_projection_model.get_layer('prototype').set_weights(tf.transpose(w))

                    for i, inputs in enumerate(dataset):
                        loss = train_step_fn(inputs)
                        step_wise_loss.append(loss)

                        self.training_logger.debug('training step: {}, step_loss: {:.3f};'.format(i + 1, loss))

                    epoch_loss = np.mean(step_wise_loss)
                    epoch_wise_loss.append(epoch_loss)

                    self.training_logger.info('epoch: {}, epoch_loss: {:.3f};'.format(epoch + 1, epoch_loss))

                    callback.on_epoch_end(epoch, logs={'loss': epoch_loss})

                self.training_logger.info('SwAV Training is completed.')
            except Exception as e:
                self.training_logger.error(e)
                raise RuntimeError(e)

        return train

    def train_step(self, global_batch_size, per_replica_batch_size):
        @tf.function
        def step(input_views):
            inputs = tf.TensorArray(dtype=tf.float32, size=5, clear_after_read=False, infer_shape=False)
            for index in range(5):
                inputs = inputs.write(index, input_views[index])

            # e.g. [2, 3] -> needed for grouping same size crops so that embeddings
            unique_consecutive_count = tf.unique_with_counts(self.crop_sizes).count
            # e.g. [2, 5] -> needed for slicing inputs to include grouped crops
            consecutive_crops = tf.cumsum(unique_consecutive_count)
            # multi-res forward passes
            start_idx = 0

            with tf.GradientTape() as tape:
                embeddings = tf.TensorArray(dtype=tf.float32, size=2, infer_shape=False)

                for idx_consecutive_crops in range(2):
                    with tape.stop_recording():
                        end_idx = consecutive_crops[idx_consecutive_crops]
                        concat_reshape = self.current_crop_size(per_replica_batch_size, start_idx)
                        # need to reshape because inputs.gather would add an extra dimension.
                        concat_input = tf.reshape(inputs.gather(tf.range(start_idx, end_idx)),
                                                  shape=concat_reshape)
                        start_idx = end_idx
                    _embedding = self.feature_backbone_model(concat_input)  # get embedding of same dim views together
                    embeddings = embeddings.write(idx_consecutive_crops, _embedding)

                projection, prototype = self.prototype_projection_model(embeddings.concat())
                _ = tf.stop_gradient(projection)

                loss = 0.0
                # Swap Assigment only happens on 224x224 crops
                # crops_for_assign = [0, 1] -> first and second crops are 224x224
                for idx_crop_for_assign in range(2):
                    with tape.stop_recording():
                        crop_id = self.crops_for_assign[idx_crop_for_assign]

                        # prototype slice for the current 224x224 crop
                        crop_prototype_start = per_replica_batch_size * crop_id
                        crop_prototype_end = per_replica_batch_size * (crop_id + 1)
                        crop_prototype = tf.gather(prototype, tf.range(crop_prototype_start, crop_prototype_end))
                        # Cluster assignment prediction
                        # Get cluster assignments using Sinkhorn Knopp: Optical Transport
                        # The code q is considered the "ground truth" - or a tuned prototype
                        crop_prototype_q_code = self.sinkhorn(crop_prototype)

                        # Swapped comparison initialization
                        # sum([2, 3] = 5
                        crops_interval = tf.reduce_sum(self.num_crops)
                        # [0, 1, 2, 3, 4]
                        all_crops_indexes = tf.range(crops_interval)

                        # take out one 224x224 crop and compare with the rest (Swap Assigment)
                        # [1, 2, 3, 4] or [0, 2, 3, 4]
                        crops_to_compare_indexes = tf.boolean_mask(all_crops_indexes,
                                                                   tf.not_equal(all_crops_indexes, crop_id))

                    sub_loss = 0.0

                    for crop_to_compare_index in crops_to_compare_indexes:
                        with tape.stop_recording():
                            crop_to_compare_prototype_start = per_replica_batch_size * crop_to_compare_index
                            crop_to_compare_prototype_end = per_replica_batch_size * (crop_to_compare_index + 1)

                        crop_to_compare_prototype = prototype[crop_to_compare_prototype_start:crop_to_compare_prototype_end]
                        crop_probability = tf.nn.softmax(crop_to_compare_prototype / self.temperature)
                        cross_entropy_loss = self.loss(crop_prototype_q_code, crop_probability)

                        # average loss entropy
                        # use compute_average_loss instead of reduce_mean to allow distributed training
                        sub_loss += tf.nn.compute_average_loss(cross_entropy_loss, global_batch_size=global_batch_size)

                    loss += sub_loss / tf.cast((tf.reduce_sum(self.num_crops) - 1), tf.float32)

                loss /= len(self.crops_for_assign)

            # back propagation
            variables = self.feature_backbone_model.trainable_variables + \
                self.prototype_projection_model.trainable_variables

            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            return loss

        return step

    @staticmethod
    def current_crop_size(batch_size, start_index):
        def high_res_shape_fn(): return tf.constant([batch_size * 2, -1, 224, 224, 3])

        def low_res_shape_fn(): return tf.constant([batch_size * 3, -1, 96, 96, 3])

        return tf.cond(tf.equal(start_index, 0), high_res_shape_fn, low_res_shape_fn)

    @staticmethod
    def sinkhorn(sample_prototype_batch):
        q_codes = tf.transpose(tf.exp(sample_prototype_batch / 0.05))
        q_codes /= tf.keras.backend.sum(q_codes)
        k_dim, b_dim = q_codes.shape

        r = tf.ones_like(k_dim, dtype=tf.float32) / k_dim
        c = tf.ones_like(b_dim, dtype=tf.float32) / b_dim

        for _ in range(3):
            u = tf.keras.backend.sum(q_codes, axis=1)
            q_codes *= tf.expand_dims((r / u), axis=1)
            q_codes *= tf.expand_dims(c / tf.keras.backend.sum(q_codes, axis=0), 0)

        final_quantity = q_codes / tf.keras.backend.sum(q_codes, axis=0, keepdims=True)
        final_quantity = tf.transpose(final_quantity)

        return final_quantity

    def _get_model_type(self):
        return SWAV

    @staticmethod
    def _get_loss():
        return tf.keras.losses.CategoricalCrossentropy(axis=1, reduction=tf.keras.losses.Reduction.NONE)

    def get_callback(self, checkpoint_storage_path, model):
        callback = SwAVCallback(model[0], model[1], checkpoint_storage_path)
        return callback

    def configure(self, models):
        print('SwAV does not require a configure method.')
