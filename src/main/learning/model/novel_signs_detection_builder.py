import logging
import os

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow import feature_column

from base_model_builder import BaseModelBuilder
from opticalflow_model_builder import OpticalFlowModelBuilder
from rgb_recurrent_model_builder import RGBRecurrentModelBuilder
from wrapper_layer import WrapperLayer
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, AUC, Precision
from learning.common.model_type import NSDM


class NovelSignsDetectionModelBuilder(BaseModelBuilder):
    def __init__(self, opticalflow_model: Model, rgb_model: Model):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.opticalflow_model = opticalflow_model
        self.rgb_model = rgb_model
        # self.rgb_feature_dim = self.imagenet_img_width * self.imagenet_img_height * self.rgb_no_channels
        # self.opticalflow_model_path = os.path.join(self.working_folder, OpticalFlowModel.SAVED_MODEL_FOLDER_NAME)
        # self.rgb_model_path = os.path.join(self.working_folder, RGBModel.SAVED_MODEL_FOLDER_NAME)

    def build(self) -> Model:
        opticalflow_input_shape = (self.imagenet_img_width, self.imagenet_img_height, self.rgb_no_channels)
        rgb_input_shape = (None, self.imagenet_img_width * self.imagenet_img_height * self.rgb_no_channels)

        opticalflow_input = Input(shape=opticalflow_input_shape)
        rgb_input = Input(shape=rgb_input_shape)

        return Model()

    def get_model_type(self):
        return NSDM

    # def get_dataset(self, dataset_path, training=True, **kwargs) -> tf.data.Dataset:
    #     # Obtain opticalflow images and class
    #     # from the class from optical flow look for rgb frames using the class and
    #
    #     batch_size = kwargs['BatchSize']
    #     shuffle_buffer_size = kwargs['ShuffleBufferSize']
    #     img_width = kwargs['ImageWidth']
    #     img_height = kwargs['ImageHeight']
    #     no_channels = kwargs['NoChannels']
    #
    #     opticalflow_dataset_path = os.path.join(dataset_path, OpticalFlowModelBuilder.MODEL_NAME)
    #     rgb_dataset_path = os.path.join(dataset_path, RGBRecurrentModelBuilder.MODEL_NAME)
    #
    #     # First we will need only opticalflow samples file paths
    #     dataset = tf.data.Dataset.list_files(opticalflow_dataset_path + '/*/*'). \
    #         map(lambda x: self.samples_processing(x, rgb_dataset_path, no_channels, img_width, img_height))
    #
    #     # for X, label in dataset.take(1):
    #     #     print(X.numpy().shape)
    #     #     print(X.numpy())
    #     #     print(label.numpy())
    #
    #     return dataset
    #
    # def build(self, **kwargs) -> Model:
    #     learning_rate = kwargs['LearningRate']
    #
    #     no_classes = len(self.classes)
    #
    #     # inner model
    #     # opticalflow_model = OpticalFlowModel(self.working_folder, self.dataset_root_path).get_model(**kwargs)
    #     # rgb_recurrent_model = RGBRecurrentModel(self.working_folder, self.dataset_root_path).get_model(**kwargs)
    #     # wrapper_layer = WrapperLayer(opticalflow_model, rgb_recurrent_model)
    #
    #     # label_column = feature_column.numeric_column('label')
    #     # feature_columns = feature_column.crossed_column()
    #
    #     # Opticalflow model definition
    #     inputs = Input(shape=(None, ), name='inputs')
    #     # x = feature_columns()(inputs)
    #     x = WrapperLayer(self.working_folder, self.dataset_root_path, **kwargs)(inputs)
    #     outputs = Dense(no_classes, activation='softmax', name='nSDm_Classifier')(x)
    #
    #     # Opticalflow model assembling
    #     model = Model(inputs=inputs, outputs=outputs, name='OpticalflowModel')
    #
    #     # Opticalflow model compilation
    #     optimizer = Adam(learning_rate)
    #     model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(),
    #                   metrics=[Recall(), AUC(curve='PR'), Precision()])
    #
    #     return model
    #
    # @staticmethod
    # def get_image_label(file_path):
    #     # convert the path to a list of path components
    #     parts = tf.strings.split(file_path, os.path.sep)
    #     # The second to last is the class-directory
    #     return parts[-2]
    #
    # @staticmethod
    # def get_image_index_by_label(file_path):
    #     parts = tf.strings.split(file_path, os.path.sep)
    #     file_name = tf.strings.regex_replace(parts[-1], '.jpg', '')
    #     return tf.strings.split(file_name, '-')[1]
    #
    # def samples_processing(self, opticalflow_sample_path, rgb_dataset_path, no_channels, img_width, img_height):
    #     # holds a label (e.g. casa, padres, cine)
    #     sample_label = self.get_image_label(opticalflow_sample_path)
    #     # index that helps to map the opticalflow sample with the RGB frames sample
    #     sample_label_index = self.get_image_index_by_label(opticalflow_sample_path)
    #
    #     # obtain an opticalflow sample in PIL format
    #     opticalflow_img = tf.io.read_file(opticalflow_sample_path)
    #     # normalizes the opticalflow sample from PIL format to a numeric format with values between 0 to 1
    #     decoded_opticalflow_img = self.decode_img(opticalflow_img, no_channels, img_width, img_height)
    #
    #     # tensors list for building a files pattern used to obtain RGB sample frames
    #     rgb_samples_pattern = [rgb_dataset_path, os.path.sep, sample_label, os.path.sep, sample_label_index, '/*']
    #     # holds RGB sample frame paths
    #     rgb_sample_frame_paths = tf.io.matching_files(tf.strings.join(rgb_samples_pattern))
    #
    #     # decodes rgb sample frames from PIL format to a normalized numeric format with values between 0 to 1
    #     decoded_rgb_frames_imgs = tf.map_fn(
    #         lambda x: self.decode_img(tf.io.read_file(x), no_channels, img_width, img_height),
    #         rgb_sample_frame_paths,
    #         dtype=tf.float32)
    #
    #     return decoded_opticalflow_img, decoded_rgb_frames_imgs, sample_label
    #
    # @staticmethod
    # def decode_img(img, no_channels, img_width, img_height):
    #     # convert the compressed string to a 3D uint8 tensor
    #     img = tf.image.decode_jpeg(img, channels=no_channels)
    #     # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    #     img = tf.image.convert_image_dtype(img, tf.float32)
    #     # resize the image to the desired size.
    #     return tf.image.resize(img, [img_width, img_height])
    #
    # def print_image_batch(self, image_batch, label_batch):
    #     plt.figure(figsize=(10, 10))
    #     for n in range(25):
    #         ax = plt.subplot(5, 5, n + 1)
    #         plt.imshow(image_batch[n])
    #         plt.title(self.classes[label_batch[n] == 1][0].title())
    #         plt.axis('off')
