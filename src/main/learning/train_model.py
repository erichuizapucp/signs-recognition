import os
import logging
import tensorflow as tf

from argparse import ArgumentParser
from logger_config import setup_logging

from learning.execution.legacy.opticalflow_executor import OpticalflowExecutor
from learning.execution.legacy.rgb_executor import RGBExecutor
from learning.execution.legacy.nsdm_executor import NSDMExecutor
from learning.execution.legacy.nsdm_v2_executor import NSDMExecutorV2
from learning.execution.swav.swav_executor import SwAVExecutor

from learning.dataset.prepare.legacy.rgb_dataset_preparer import RGBDatasetPreparer
from learning.dataset.prepare.legacy.opticalflow_dataset_preparer import OpticalflowDatasetPreparer
from learning.dataset.prepare.legacy.combined_dataset_preparer import CombinedDatasetPreparer
from learning.dataset.prepare.swav.swav_video_dataset_preparer import SwAVDatasetPreparer

from learning.model import models
from learning.common.model_type import OPTICAL_FLOW, RGB, NSDM, NSDMV2, SWAV
from learning.common.model_utility import ModelUtility

DEFAULT_NO_EPOCHS = 5
DEFAULT_NO_STEPS_EPOCHS = None
DEFAULT_NO_BATCH_SIZE = 64


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', help='Model Name', required=True)
    parser.add_argument('-tr', '--train_dataset_path', help='Train Dataset Path', required=True)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=DEFAULT_NO_EPOCHS)
    parser.add_argument('-ns', '--no_steps', help='Number of steps per epoch', default=DEFAULT_NO_STEPS_EPOCHS)
    parser.add_argument('-bs', '--batch_size', help='Dataset batch size', default=DEFAULT_NO_BATCH_SIZE)
    parser.add_argument('-dp', '--detect_person', help='Detect person on frames', action='store_true', required=False)
    parser.add_argument('-odm', '--person_detection_model_name', help='Person Detection Model Name', required=False)
    parser.add_argument('-odcp', '--person_detection_checkout_prefix', help='Person Detection Checkout Prefix',
                        required=False)
    parser.add_argument('-mt', '--mirrored_training', help='Use Mirrored Training', action='store_true', required=False)
    parser.add_argument('-nr', '--no_replicas', help='No Replicas', required=False, default=0)
    parser.add_argument('-gp', '--gpus', help='GPUs to use', required=False)

    return parser.parse_args()


def get_model(model_name):
    built_models = {
        OPTICAL_FLOW: lambda: models.get_opticalflow_model(),
        RGB: lambda: models.get_rgb_model(),
        NSDM: lambda: models.get_nsdm_model(),
        NSDMV2: lambda: models.get_nsdm_v2_model(),
        SWAV: lambda: models.get_swav_models()
    }

    def model_fn():
        return built_models[model_name]()

    return model_fn


def get_distributed_model(distribute_strategy, get_model_fn):
    with distribute_strategy.scope():
        model = get_model_fn()
    return model


def get_dataset(model_name, train_dataset_path, batch_size, **kwargs):
    detect_person = kwargs['detect_person'] if 'detect_person' in kwargs else False
    person_detection_model = None
    if detect_person:
        person_detection_model_name = kwargs['person_detection_model_name']
        person_detection_checkout_prefix = kwargs['person_detection_checkout_prefix']

        model_utility = ModelUtility()
        person_detection_model = model_utility.get_object_detection_model(person_detection_model_name,
                                                                          person_detection_checkout_prefix)
    data_preparers = {
        OPTICAL_FLOW: lambda: OpticalflowDatasetPreparer(train_dataset_path, test_dataset_path=None),
        RGB: lambda: RGBDatasetPreparer(train_dataset_path, test_dataset_path=None),
        NSDM: lambda: CombinedDatasetPreparer(train_dataset_path, test_dataset_path=None),
        NSDMV2: lambda: CombinedDatasetPreparer(train_dataset_path, test_dataset_path=None),
        SWAV: lambda: SwAVDatasetPreparer(train_dataset_path, test_dataset_path=None,
                                          person_detection_model=person_detection_model),
    }
    data_preparer = data_preparers[model_name]()

    def prepare_dataset():
        return data_preparer.prepare_train_dataset(batch_size)

    return prepare_dataset


def get_distributed_dataset(distribute_strategy, get_dataset_fn):
    return distribute_strategy.experimental_distribute_dataset(get_dataset_fn())


def get_executor(executor_name, models_to_execute):
    executors = {
        OPTICAL_FLOW: lambda: OpticalflowExecutor(),
        RGB: lambda: RGBExecutor(),
        NSDM: lambda: NSDMExecutor(),
        NSDMV2: lambda: NSDMExecutorV2(),
        SWAV: lambda: SwAVExecutor(),
    }
    executor = executors[executor_name]()

    if executor_name != SWAV:
        executor.configure(models_to_execute)

    return executor


def get_distributed_strategy(no_replicas: 0, logger):
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        logger.debug('Using GPU Training with %s replicas', no_replicas if no_replicas > 0 else len(gpus))
        gpu_device_names = ["GPU:{}".format(replica_id) for replica_id in range(no_replicas)]
        return tf.distribute.MirroredStrategy(devices=gpu_device_names) \
            if no_replicas > 0 else tf.distribute.MirroredStrategy()
    else:
        cpus = tf.config.list_physical_devices('CPU')
        logger.debug('Using CPU Training with %s replicas', no_replicas if no_replicas > 0 else len(gpus))
        if len(cpus) == 1 and no_replicas:
            tf.config.set_logical_device_configuration(cpus[0],
                                                       [tf.config.LogicalDeviceConfiguration()] * int(no_replicas))
            logical_gpus = tf.config.list_logical_devices('CPU')
            return tf.distribute.MirroredStrategy(logical_gpus)
        else:
            cpu_device_names = ["CPU:{}".format(replica_id) for replica_id in range(no_replicas)]
            return tf.distribute.MirroredStrategy(devices=cpu_device_names) \
                if no_replicas > 0 else tf.distribute.MirroredStrategy()


def get_distributed_train_step(distribute_strategy, train_step_fn):
    @tf.function
    def step(inputs):
        per_replica_losses = distribute_strategy.run(train_step_fn, args=(inputs,))
        return distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    return step


def get_distributed_optimizer(distribute_strategy, get_optimizer_fn):
    with distribute_strategy.scope():
        optimizer = get_optimizer_fn()
        return optimizer


def get_distributed_callback(distribute_strategy, get_callback_fn):
    with distribute_strategy.scope():
        callback = get_callback_fn()
        return callback


def main():
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'learning-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()
    no_epochs = int(args.no_epochs)
    no_steps_per_epoch = args.no_steps
    batch_size_per_replica = int(args.batch_size)

    model_name = args.model
    train_dataset_path = args.train_dataset_path
    executor_name = model_name

    detect_person = args.detect_person
    person_detection_model_name = args.person_detection_model_name
    person_detection_checkout_prefix = args.person_detection_checkout_prefix

    mirrored_training = args.mirrored_training
    no_replicas = int(args.no_replicas)
    gpus = args.gpus

    if gpus and len(gpus) > 0:
        set_visible_gpus(gpus.split(','))

    distribute_strategy = get_distributed_strategy(no_replicas, logger) if mirrored_training else None
    batch_size = batch_size_per_replica * \
        distribute_strategy.num_replicas_in_sync if mirrored_training else batch_size_per_replica

    logger.debug('learning operation started with the following parameters: %s', args)

    get_model_fn = get_model(model_name)
    model = get_distributed_model(distribute_strategy, get_model_fn) if mirrored_training else get_model_fn()

    executor = get_executor(executor_name, model)

    if model_name == SWAV:
        # Custom training loops require a distributed dataset to be passed
        get_dataset_fn = get_dataset(model_name,
                                     train_dataset_path,
                                     batch_size,
                                     detect_person=detect_person,
                                     person_detection_model_name=person_detection_model_name,
                                     person_detection_checkout_prefix=person_detection_checkout_prefix)

        dataset = get_distributed_dataset(distribute_strategy,
                                          get_dataset_fn) if mirrored_training else get_dataset_fn()
        optimizer = get_distributed_optimizer(distribute_strategy,
                                              executor.get_optimizer) if mirrored_training else executor.get_optimizer()
        callback = get_distributed_callback(distribute_strategy,
                                            executor.get_callback) if mirrored_training else executor.get_callback()
        train_step = get_distributed_train_step(distribute_strategy, executor.train_step(batch_size,
                                                                                         batch_size_per_replica)) if \
            mirrored_training else executor.train_step(batch_size, batch_size_per_replica)
        executor.train_model(model, dataset, no_epochs, no_steps_per_epoch)(train_step, optimizer, callback)
    else:
        # Keras model.fit loops don't require a distributed dataset to be passed
        get_dataset_fn = get_dataset(model_name, train_dataset_path, batch_size, detect_person=detect_person)
        dataset = get_dataset_fn()

        executor.train_model(model, dataset, no_epochs, no_steps_per_epoch)

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
