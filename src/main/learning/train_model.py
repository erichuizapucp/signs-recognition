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


def get_cmd_args():
    parser = ArgumentParser(description="Peruvian Signs Language Self Supervised Learning")

    # General arguments
    parser.add_argument('--model', type=str, help='Model Name', required=True)
    parser.add_argument('--train_dataset_path', type=str, help='Train Dataset Path', required=True)
    parser.add_argument('--model_storage_path', type=str, help='Model Storage Path', required=True)
    parser.add_argument('--checkpoint_storage_path', type=str, help='Checkpoint Storage Path', required=True)
    parser.add_argument('--failure_reason_path', type=str, help='Failure Reason', required=True)

    # General training arguments
    parser.add_argument('--no_epochs', type=int, help='Number of epochs', default=20)
    parser.add_argument('--no_steps', type=int, help='Number of steps per epoch', default=1000)
    parser.add_argument('--batch_size', type=int, help='Dataset batch size', default=3)
    parser.add_argument('--start_learning_rate', type=float, help='Start Learning Rate', default=4.8)
    parser.add_argument('--end_learning_rate', type=float, help='End Learning Rate', default=0.0001)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--clip_value', type=float, help='Clip Value', default=0.25)

    # SwAV specific arguments
    parser.add_argument('--num_crops', type=int, help='SwAV number of multi-crops', nargs='+', default=[2, 3])
    parser.add_argument('--crops_for_assign', type=int, help='SwAV crops for assign', nargs='+', default=[0, 1])
    parser.add_argument('--crop_sizes_list', type=int, help='SwAV crop sizes list', nargs='+',
                        default=[224, 224, 96, 96, 96])
    parser.add_argument('--crop_sizes', type=int, help='SwAV crop sizes', nargs='+', default=[224, 96])
    parser.add_argument('--min_scale', type=float, help='SwAV Multi-crop min scale', nargs='+', default=[0.14, 0.05])
    parser.add_argument('--max_scale', type=float, help='SwAV Multi-crop max scale', nargs='+', default=[1., 0.14])
    parser.add_argument('--sample_duration_range', type=float, help='', nargs='+', default=[0.5, 1.0])

    # SwAV models specific arguments
    parser.add_argument('--temperature', type=float, help='SwAV temperature', nargs='+', default=0.1)
    parser.add_argument('--no_projection_1_neurons', type=int, help='Dense Layer 1 inputs size', default=896)
    parser.add_argument('--no_projection_2_neurons', type=int, help='Dense Layer 2 inputs size', default=768)
    parser.add_argument('--prototype_vector_dim', type=int, help='Prototype vector size', default=512)
    parser.add_argument('--lstm_cells', type=int, help='Number of LSTM cells', default=512)
    parser.add_argument('--embedding_size', type=int, help='Embedding size', default=1024)
    parser.add_argument('--l2_regularization_epsilon', type=float, help='L2 regularization epsilon', default=0.05)

    # Mirrored training arguments
    parser.add_argument('--mirrored_training', type=bool, help='Use Mirrored Training', default=False)
    parser.add_argument('--no_replicas', type=int, help='No Replicas', default=4)

    # Person detection arguments
    parser.add_argument('--person_detection_model_name', help='Person Detection Model Name',
                        default='centernet_resnet50_v1_fpn_512x512_coco17_tpu-8')
    parser.add_argument('--person_detection_checkout_prefix', help='Person Detection Checkout Prefix', default='ckpt-0')

    return parser.parse_args()


def get_model(args):
    built_models = {
        OPTICAL_FLOW: lambda: models.get_opticalflow_model(),
        RGB: lambda: models.get_rgb_model(),
        NSDM: lambda: models.get_nsdm_model(),
        NSDMV2: lambda: models.get_nsdm_v2_model(),
        SWAV: lambda: models.get_swav_models(no_projection_1_neurons=args.no_projection_1_neurons,
                                             no_projection_2_neurons=args.no_projection_2_neurons,
                                             prototype_vector_dim=args.prototype_vector_dim,
                                             lstm_cells=args.lstm_cells,
                                             embedding_size=args.embedding_size,
                                             l2_regularization_epsilon=args.l2_regularization_epsilon)
    }

    def model_fn():
        return built_models[args.model]()

    return model_fn


def get_distributed_model(distribute_strategy, get_model_fn):
    with distribute_strategy.scope():
        model = get_model_fn()
    return model


def get_dataset(model_name, train_dataset_path, batch_size, **kwargs):
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
        SWAV: lambda: SwAVDatasetPreparer(train_dataset_path,
                                          test_dataset_path=None,
                                          person_detection_model=person_detection_model,
                                          crop_sizes=kwargs['crop_sizes'],
                                          num_crops=kwargs['num_crops'],
                                          min_scale=kwargs['min_scale'],
                                          max_scale=kwargs['max_scale'],
                                          sample_duration_range=kwargs['sample_duration_range'])
    }
    data_preparer = data_preparers[model_name]()

    def prepare_dataset():
        return data_preparer.prepare_train_dataset(batch_size)

    return prepare_dataset


def get_distributed_dataset(distribute_strategy, get_dataset_fn):
    return distribute_strategy.experimental_distribute_dataset(get_dataset_fn())


def get_executor(models_to_execute, args):
    executors = {
        OPTICAL_FLOW: lambda: OpticalflowExecutor(start_learning_rate=args.start_learning_rate,
                                                  momentum=args.momentum,
                                                  clip_value=args.clip_value),

        RGB: lambda: RGBExecutor(start_learning_rate=args.start_learning_rate,
                                 momentum=args.momentum,
                                 clip_value=args.clip_value),

        NSDM: lambda: NSDMExecutor(start_learning_rate=args.start_learning_rate,
                                   momentum=args.momentum,
                                   clip_value=args.clip_value),

        NSDMV2: lambda: NSDMExecutorV2(start_learning_rate=args.start_learning_rate,
                                       momentum=args.momentum,
                                       clip_value=args.clip_value),

        SWAV: lambda: SwAVExecutor(start_learning_rate=args.start_learning_rate,
                                   momentum=args.momentum,
                                   clip_value=args.clip_value,
                                   end_learning_rate=args.end_learning_rate,
                                   num_crops=args.num_crops,
                                   crops_for_assign=args.crops_for_assign,
                                   crop_sizes_list=args.crop_sizes_list,
                                   temperature=args.temperature),
    }
    executor = executors[args.model]()

    if args.model != SWAV:
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


def get_distributed_optimizer(distribute_strategy, get_optimizer_fn, no_epochs, no_steps):
    with distribute_strategy.scope():
        optimizer = get_optimizer_fn(no_epochs, no_steps)
        return optimizer


def get_distributed_callback(distribute_strategy, get_callback_fn, checkpoint_storage_path, model):
    with distribute_strategy.scope():
        callback = get_callback_fn(checkpoint_storage_path, model)
        return callback


def main():
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'learning-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()

    distribute_strategy = get_distributed_strategy(args.no_replicas, logger) if args.mirrored_training else None
    batch_size = args.batch_size * \
        distribute_strategy.num_replicas_in_sync if args.mirrored_training else args.batch_size

    logger.debug('learning operation started with the following parameters: %s', args)

    get_model_fn = get_model(args)
    model = get_distributed_model(distribute_strategy, get_model_fn) if args.mirrored_training else get_model_fn()

    executor = get_executor(model, args)

    if args.model == SWAV:
        # Custom training loops require a distributed dataset to be passed
        get_dataset_fn = get_dataset(args.model,
                                     args.train_dataset_path,
                                     batch_size,
                                     person_detection_model_name=args.person_detection_model_name,
                                     person_detection_checkout_prefix=args.person_detection_checkout_prefix,
                                     crop_sizes=args.crop_sizes,
                                     num_crops=args.num_crops,
                                     min_scale=args.min_scale,
                                     max_scale=args.max_scale,
                                     sample_duration_range=args.sample_duration_range)

        dataset = get_distributed_dataset(distribute_strategy, get_dataset_fn) if args.mirrored_training else get_dataset_fn()
        optimizer = get_distributed_optimizer(distribute_strategy, executor.get_optimizer, args.no_epochs, args.no_steps) \
            if args.mirrored_training \
            else \
            executor.get_optimizer(args.no_epochs, args.no_steps)

        # callback assignation
        callback = get_distributed_callback(distribute_strategy, executor.get_callback, args.checkpoint_storage_path, model) \
            if args.mirrored_training \
            else \
            executor.get_callback(args.checkpoint_storage_path, model)

        # training step function assignation
        train_step_fn = get_distributed_train_step(distribute_strategy, executor.train_step(batch_size, args.batch_size)) \
            if args.mirrored_training \
            else \
            executor.train_step(batch_size, args.batch_size)

        executor.train_model(model,
                             dataset,
                             args.no_epochs,
                             args.no_steps,
                             model_storage_path=args.model_storage_path,
                             failure_storege_path=args.failure_reason_path)(train_step_fn, optimizer, callback)
    else:
        # Keras model.fit loops don't require a distributed dataset to be passed
        get_dataset_fn = get_dataset(args.model, args.train_dataset_path, batch_size)
        dataset = get_dataset_fn()

        executor.train_model(model, dataset, args.no_epochs, args.no_steps)

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
