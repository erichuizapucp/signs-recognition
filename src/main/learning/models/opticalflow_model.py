import logging
import tensorflow as tf
import os

from models.base_model import BaseModel

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, AUC, Precision


class OpticalFlowModel(BaseModel):
    SAVED_MODEL_FOLDER_NAME = 'saved-models/opticalflow/'

    def __init__(self, working_folder, dataset_path, **kwargs):
        super(OpticalFlowModel, self).__init__(working_folder, dataset_path, **kwargs)

        self.logger = logging.getLogger(__name__)

        self.dataset: tf.data.Dataset = self.get_dataset()
        self.model: Model = self.get_model()
        self.model_history = None

    def get_model(self) -> Model:
        # use a ResNet152 pre trained model as a base model, we are not including the top layer for maximizing the
        # learned features
        pre_trained_model: Model = ResNet152V2(include_top=False, weights='imagenet')
        # Freeze pre-trained base model weights
        pre_trained_model.trainable = False

        no_classes = len(self.classes)

        # Opticalflow model definition
        inputs = Input(shape=(self.img_width, self.img_height, self.no_channels), name='inputs')
        x = pre_trained_model(inputs)
        x = GlobalAveragePooling2D(name='OpticalflowGlobalAvgPooling')(x)
        outputs = Dense(no_classes, activation='softmax', name='OpticalflowClassifier')(x)

        # Opticalflow model assembling
        model = Model(inputs=inputs, outputs=outputs, name='OpticalflowModel')
        self.logger.debug('Opticalflow model summary: \n %s', model.summary())

        # Opticalflow model compilation
        optimizer = Adam(self.learning_rate)
        model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=[Recall(), AUC(), Precision()])

        return model

    def get_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.list_files(self.dataset_path + '/*/*')
        labeled_dataset = dataset.map(self.__process_image_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return labeled_dataset

    def train(self):
        self.logger.debug('Opticalflow model train started with the following parameters')
        self.model_history = self.model.fit(self.dataset, epochs=self.no_epochs, batch_size=self.batch_size)

        # save trained model and weights to the file system for future use
        self.__save_model()

    def evaluate(self):
        self.logger.debug('Opticalflow model evaluation started')
        self.model.evaluate()

    def predict(self):
        self.logger.debug('Opticalflow model predict started')
        self.model.predict(self.dataset)

    def __get_image_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == self.classes

    def __decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=self.no_channels)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.img_width, self.img_height])

    def __process_image_path(self, file_path):
        label = self.__get_image_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.__decode_img(img)
        return img, label
