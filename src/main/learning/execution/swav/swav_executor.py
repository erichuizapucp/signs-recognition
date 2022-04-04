import numpy as np
import tensorflow as tf

from learning.execution.base_model_executor import BaseModelExecutor

from itertools import groupby
from learning.execution.swav.swav_callback import SwAVCallback
from learning.common.model_utility import SWAV


class SwAVExecutor(BaseModelExecutor):
    def __init__(self):
        super().__init__()

        # 224x224, 224x224, 96x96, 96x96, 96x96
        self.num_crops = [2, 3]

        # swapped assigment is intended only for 224x224 crops
        self.crops_for_assign = [0, 1]
        self.temperature = 0.1

        # To be assigned on the train_model method
        self.optimizer = None
        self.feature_backbone_model = None
        self.prototype_projection_model = None

        self.loss = self._get_loss()

        self.training_logger.info('SwAV initialized with: num_crops: %s, crops_for_assign: %s, '
                                  'temperature: %s', self.num_crops, self.crops_for_assign, self.temperature)

    def train_model(self, models, dataset, no_epochs, no_steps_per_epoch=None):
        def train(optimizer, train_step):
            try:
                self.feature_backbone_model = models[0]
                self.prototype_projection_model = models[1]
                self.optimizer = optimizer

                callback = SwAVCallback(self.feature_backbone_model, self.prototype_projection_model)

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
                        loss = train_step(inputs)
                        step_wise_loss.append(loss)

                        self.training_logger.debug('training step: {} loss: {:.3f}'.format(i + 1, loss))

                    epoch_loss = np.mean(step_wise_loss)
                    epoch_wise_loss.append(epoch_loss)

                    self.training_logger.info('epoch: {} loss: {:.3f}'.format(epoch + 1, epoch_loss))

                    callback.on_epoch_end(epoch, logs={'loss': epoch_loss})

                self.training_logger.info('SwAV Training is completed.')
            except Exception as e:
                self.training_logger.error(e)

        return train

    def train_step(self, global_batch_size):
        def step(input_views):
            clip1, clip2, clip3, clip4, clip5 = input_views
            inputs = [clip1, clip2, clip3, clip4, clip5]
            per_replica_batch_size = inputs[0].shape[0]

            # list of crop size of views e.g. [224, 224, 96, 96, 96]
            crop_sizes = [inp.shape[2] for inp in inputs]

            # e.g. [2, 3] -> needed for grouping same size crops so that embeddings
            # for the same can be calculated
            unique_consecutive_count = [len([crop_elem for crop_elem in group]) for _, group in groupby(crop_sizes)]

            # e.g. [2, 5] -> needed for slicing inputs to include grouped crops
            idx_crops = tf.cumsum(unique_consecutive_count)

            # multi-res forward passes
            start_idx = 0
            with tf.GradientTape() as tape:
                for end_idx in idx_crops:
                    concat_input = tf.stop_gradient(tf.concat(inputs[start_idx:end_idx], axis=0))
                    _embedding = self.feature_backbone_model(concat_input)  # get embedding of same dim views together
                    if start_idx == 0:
                        embeddings = _embedding  # for first iter
                    else:
                        # concat all the embeddings from all the views
                        embeddings = tf.concat((embeddings, _embedding), axis=0)
                    start_idx = end_idx

                # get normalized projection and prototype
                projection, prototype = self.prototype_projection_model(embeddings)
                _ = tf.stop_gradient(projection)

                loss = 0
                # Swap Assigment only happens on 224x224 crops
                # crops_for_assign = [0, 1] -> first and second crops are 224x224
                for i, crop_id in enumerate(self.crops_for_assign):
                    with tape.stop_recording():
                        # prototype slice for the current 224x224 crop
                        crop_prototype_start = per_replica_batch_size * crop_id
                        crop_prototype_end = per_replica_batch_size * (crop_id + 1)
                        crop_prototype = prototype[crop_prototype_start: crop_prototype_end]

                        # Cluster assignment prediction
                        # Get cluster assignments using Sinkhorn Knopp: Optical Transport
                        # The code q is considered the "ground truth" - or a tuned prototype
                        crop_prototype_q_code = self.sinkhorn(crop_prototype)

                        # Swapped comparison initialization
                        # sum([2, 3] = 5
                        crops_interval = int(np.sum(self.num_crops))
                        # [0, 1, 2, 3, 4]
                        all_crops_indexes = np.arange(crops_interval)

                        # take out one 224x224 crop and compare with the rest (Swap Assigment)
                        # [1, 2, 3, 4] or [0, 2, 3, 4]
                        crops_to_compare_indexes = np.delete(all_crops_indexes, crop_id)

                    sub_loss = 0

                    for crop_to_compare_index in crops_to_compare_indexes:
                        with tape.stop_recording():
                            crop_to_compare_prototype_start = per_replica_batch_size * crop_to_compare_index
                            crop_to_compare_prototype_end = per_replica_batch_size * (crop_to_compare_index + 1)

                        crop_to_compare_prototype = prototype[crop_to_compare_prototype_start:
                                                              crop_to_compare_prototype_end]
                        crop_probability = tf.nn.softmax(crop_to_compare_prototype / self.temperature)
                        cross_entropy_loss = self.loss(crop_prototype_q_code, crop_probability)

                        # average loss entropy
                        # use compute_average_loss instead of reduce_mean to allow distributed training
                        sub_loss += tf.nn.compute_average_loss(cross_entropy_loss, global_batch_size=global_batch_size)

                    loss += sub_loss / tf.cast((tf.reduce_sum(self.num_crops) - 1), tf.float32)

                loss /= len(self.crops_for_assign)

            # back propagation
            variables = self.feature_backbone_model.trainable_variables + self.prototype_projection_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            return loss

        return step

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

    def configure(self, models):
        print('SwAV does not require a configure method.')
