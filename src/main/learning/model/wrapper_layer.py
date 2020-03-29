import tensorflow as tf

from tensorflow.keras.layers import Layer
from model.opticalflow_model_builder import OpticalFlowModelBuilder
from model.rgb_recurrent_builder import RGBRecurrentModel


class WrapperLayer(Layer):
    def __init__(self, working_folder, dataset_root_path, **kwargs):
        super(WrapperLayer, self).__init__()

        self.working_folder = working_folder
        self.dataset_root_path = dataset_root_path

        self.img_width = kwargs['ImageWidth']
        self.img_height = kwargs['ImageHeight']
        self.no_channels = kwargs['NoChannels']
        self.learning_rate = kwargs['LearningRate']

        self.opticalflow = OpticalFlowModelBuilder(working_folder, dataset_root_path)
        self.recurrent_rgb = RGBRecurrentModel(working_folder, dataset_root_path)

        self.opticalflow_model = None
        self.rgb_recurrent_model = None

    def build(self, input_shape):
        self.opticalflow_model = self.opticalflow.get_model(
            ImageWidth=self.img_width,
            ImageHeight=self.img_height,
            NoChannels=self.no_channels,
            LearningRate=self.learning_rate
        )

        self.rgb_recurrent_model = self.recurrent_rgb.get_model(
            ImageWidth=self.img_width,
            ImageHeight=self.img_height,
            NoChannels=self.no_channels,
            LearningRate=self.learning_rate
        )

    def call(self, inputs, **kwargs):
        return tf.matmul(input, 0)
