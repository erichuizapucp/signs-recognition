import logging
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from models.base_model import BaseModel

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, AUC, Precision


class OpticalFlowModel(BaseModel):
    MODEL_NAME = 'opticalflow'

    def __init__(self, working_folder, dataset_root_path):
        super(OpticalFlowModel, self).__init__(working_folder, dataset_root_path)

        self.logger = logging.getLogger(__name__)

    def get_dataset(self, dataset_path, **kwargs) -> tf.data.Dataset:
        batch_size = kwargs['BatchSize']
        shuffle_buffer_size = kwargs['ShuffleBufferSize']
        img_width = kwargs['ImageWidth']
        img_height = kwargs['ImageHeight']
        no_channels = kwargs['NoChannels']

        dataset = tf.data.Dataset.list_files(dataset_path + '/*/*')\
            .shuffle(buffer_size=shuffle_buffer_size)\
            .map(lambda x: self.process_image_path(x, no_channels, img_width, img_height),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .batch(batch_size)

        return dataset

    def get_model(self, **kwargs) -> Model:
        img_width = kwargs['ImageWidth']
        img_height = kwargs['ImageHeight']
        no_channels = kwargs['NoChannels']
        learning_rate = kwargs['LearningRate']

        no_classes = len(self.classes)

        # use a ResNet152 pre trained model as a base model, we are not including the top layer for maximizing the
        # learned features
        backbone_model: Model = ResNet152V2(include_top=False, weights='imagenet')
        # Freeze pre-trained base model weights
        backbone_model.trainable = False

        # Opticalflow model definition
        inputs = Input(shape=(img_width, img_height, no_channels), name='inputs')
        x = backbone_model(inputs)
        x = GlobalAveragePooling2D(name='OpticalflowGlobalAvgPooling')(x)
        outputs = Dense(no_classes, activation='softmax', name='OpticalflowClassifier')(x)

        # Opticalflow model assembling
        model = Model(inputs=inputs, outputs=outputs, name='OpticalflowModel')
        # self.logger.debug('Opticalflow model summary: \n %s', model.summary())

        # Opticalflow model compilation
        optimizer = Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(),
                      metrics=[Recall(), AUC(curve='PR'), Precision()])

        return model

    def get_image_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == self.classes

    @staticmethod
    def decode_img(img, no_channels, img_width, img_height):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=no_channels)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [img_width, img_height])

    def process_image_path(self, file_path, no_channels, img_width, img_height):
        label = self.get_image_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, no_channels, img_width, img_height)
        return img, label

    def print_image_batch(self, image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n])
            plt.title(self.classes[label_batch[n] == 1][0].title())
            plt.axis('off')
