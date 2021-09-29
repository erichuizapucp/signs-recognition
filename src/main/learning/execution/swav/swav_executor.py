import tensorflow as tf
import numpy as np

from learning.execution.base_model_executor import BaseModelExecutor
from learning.dataset.prepare.swav.swav_video_dataset_preparer import SwAVDatasetPreparer

from itertools import groupby
from tensorflow.keras.models import Model
from learning.common.model_utility import SWAV


class SwAVExecutor(BaseModelExecutor):
    def __init__(self, feature_detection_model: Model,
                 projection_model: Model,
                 train_dataset_path,
                 test_dataset_path,
                 object_detection_model):

        super().__init__(feature_detection_model, projection_model)

        self.dataset_preparer = SwAVDatasetPreparer(train_dataset_path, test_dataset_path, object_detection_model)

        self.num_crops = [2, 3]
        self.crops_for_assign = [0, 1]
        self.temperature = 0.1

    def train_model(self, no_epochs, no_steps_per_epoch=None):
        step_wise_loss = []
        epoch_wise_loss = []

        feature_backbone_model = self.__get_feature_backbone_model()
        prototype_projection_model = self.__get_prototype_projection_model()

        dataset = self._get_train_dataset()

        for epoch in range(no_epochs):
            w = prototype_projection_model.get_layer('prototype').get_weights()
            w = tf.transpose(w)
            w = tf.math.l2_normalize(w, axis=1)
            prototype_projection_model.get_layer('prototype').set_weights(tf.transpose(w))

            for i, inputs in enumerate(dataset):
                loss = self.train_step(inputs,
                                       feature_backbone_model,  # feature learning model
                                       prototype_projection_model,  # prototype projection model
                                       self._get_optimizer(),
                                       self.crops_for_assign,
                                       self.temperature)

                step_wise_loss.append(loss)
            epoch_wise_loss.append(np.mean(step_wise_loss))

            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

        return epoch_wise_loss, [feature_backbone_model, prototype_projection_model]

    def train_step(self, input_views, feature_backbone, projection_prototype, optimizer, crops_for_assign, temperature):
        clip1, clip2, clip3, clip4, clip5 = input_views
        inputs = [clip1, clip2, clip3, clip4, clip5]
        batch_size = inputs[0].shape[0]

        crop_sizes = [inp.shape[1] for inp in inputs]  # list of crop size of views
        unique_consecutive_count = [len([elem for elem in g]) for _, g in groupby(crop_sizes)]
        idx_crops = tf.cumsum(unique_consecutive_count)

        # multi-res forward passes
        start_idx = 0
        with tf.GradientTape() as tape:
            for end_idx in idx_crops:
                concat_input = tf.stop_gradient(tf.concat(inputs[start_idx:end_idx], axis=0))
                _embedding = feature_backbone(concat_input)  # get embedding of same dim views together
                if start_idx == 0:
                    embeddings = _embedding  # for first iter
                else:
                    # concat all the embeddings from all the views
                    embeddings = tf.concat((embeddings, _embedding), axis=0)
                start_idx = end_idx

            projection, prototype = projection_prototype(embeddings)  # get normalized projection and prototype
            _ = tf.stop_gradient(projection)

            loss = 0
            for i, crop_id in enumerate(crops_for_assign):  # crops_for_assign = [0,1]
                with tape.stop_recording():
                    out = prototype[batch_size * crop_id: batch_size * (crop_id + 1)]

                    # get assignments
                    q = self.sinkhorn(out)  # sinkhorn is used for cluster assignment

                # cluster assignment prediction
                sub_loss = 0
                for v in np.delete(np.arange(int(np.sum(self.num_crops))), crop_id):
                    p = tf.nn.softmax(prototype[batch_size * v: batch_size * (v + 1)] / temperature)
                    sub_loss -= tf.math.reduce_mean(tf.math.reduce_sum(q * tf.math.log(p), axis=1))
                loss += sub_loss / tf.cast((tf.reduce_sum(self.num_crops) - 1), tf.float32)

            loss /= len(crops_for_assign)

        # back propagation
        variables = feature_backbone.trainable_variables + projection_prototype.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss

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

    def __get_feature_backbone_model(self):
        return self.model1

    def __get_prototype_projection_model(self):
        return self.model2

    def configure(self):
        print('SwAV models do not require configuration.')
