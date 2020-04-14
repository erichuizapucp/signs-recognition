import os
import logging
import tensorflow as tf

from learning.dataset.tfrecord.tf_record_dataset_reader import TFRecordDatasetReader
from learning.common.imagenet_config import IMAGENET_CONFIG


class BaseDatasetPreparer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.working_dir = os.getenv('WORK_DIR', './')
        self.dataset_dir = 'serialized-dataset'

        self.imagenet_img_width = IMAGENET_CONFIG['imagenet_img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['imagenet_img_height']
        self.rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

        self.dataset_batch_size = 64

    def prepare_train_dataset(self):
        return self._prepare(self._get_train_dataset_path)

    def prepare_test_dataset(self):
        return self._prepare(self._get_test_dataset_path)

    def _prepare(self, get_dataset_path_func):
        dataset_path = get_dataset_path_func()
        dataset_reader = TFRecordDatasetReader(self._get_dataset_type(), dataset_path)
        dataset = dataset_reader.read()
        return dataset

    def _get_train_dataset_path(self):
        return os.path.join(self.working_dir, self.dataset_dir, self._get_dataset_type())

    # TODO: for now the test dataset path is the same as the train dataset path
    def _get_test_dataset_path(self):
        return self._get_train_dataset_path()

    def _transform_image(self, img):
        # convert to integers
        img = tf.image.decode_jpeg(img, channels=self.rgb_no_channels)
        # convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.imagenet_img_width, self.imagenet_img_height])

    def _prepare_sample(self, sample, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def _get_dataset_type(self):
        raise NotImplementedError('_get_dataset_type method not implemented.')
