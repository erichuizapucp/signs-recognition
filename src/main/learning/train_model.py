import os
import logging
import tensorflow as tf

from argparse import ArgumentParser
from logger_config import setup_logging

from learning.execution.legacy.opticalflow_executor import OpticalflowExecutor
from learning.execution.legacy.rgb_executor import RGBExecutor
from learning.execution.legacy.nsdm_executor import NSDMExecutor
from learning.execution.legacy.nsdm_v2_executor import NSDMExecutorV2
from learning.execution.nsdmv3.nsdm_v3_executor import NSDMExecutorV3
from learning.execution.swav.swav_executor import SwAVExecutor

from learning.dataset.prepare.legacy.rgb_dataset_preparer import RGBDatasetPreparer
from learning.dataset.prepare.legacy.opticalflow_dataset_preparer import OpticalflowDatasetPreparer
from learning.dataset.prepare.legacy.combined_dataset_preparer import CombinedDatasetPreparer
from learning.dataset.prepare.swav.serialized_swav_video_dataset_preparer import SerializedSwAVDatasetPreparer

from learning.common.model_type import OPTICAL_FLOW, RGB, NSDM, NSDMV2, NSDMV3, SWAV


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
    parser.add_argument('--start_learning_rate', type=float, help='Start Learning Rate', default=0.1)
    parser.add_argument('--end_learning_rate', type=float, help='End Learning Rate', default=0.0001)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--clip_value', type=float, help='Clip Value', default=0.25)
    parser.add_argument('--load_backbone_weights', type=bool, help='Load backbone weights', default=True)

    # SwAV specific arguments
    parser.add_argument('--num_crops', type=int, help='SwAV number of multi-crops', nargs='+', default=[2, 3])
    parser.add_argument('--crops_for_assign', type=int, help='SwAV crops for assign', nargs='+', default=[0, 1])
    parser.add_argument('--crop_sizes_list', type=int, help='SwAV crop sizes list', nargs='+',
                        default=[224, 224, 96, 96, 96])

    # SwAV models specific arguments
    parser.add_argument('--temperature', type=float, help='SwAV temperature', nargs='+', default=0.1)
    parser.add_argument('--no_projection_1_neurons', type=int, help='Dense Layer 1 inputs size', default=256)
    parser.add_argument('--no_projection_2_neurons', type=int, help='Dense Layer 2 inputs size', default=128)
    parser.add_argument('--prototype_vector_dim', type=int, help='Prototype vector size', default=64)
    parser.add_argument('--lstm_cells', type=int, help='Number of LSTM cells', default=512)
    parser.add_argument('--embedding_size', type=int, help='Embedding size', default=512)
    parser.add_argument('--l2_regularization_epsilon', type=float, help='L2 regularization epsilon', default=0.05)

    # NSDMV3 model specific arguments
    parser.add_argument('--load_weights', type=bool, help='SwAV Features Weights Path', required=False, default=False)
    parser.add_argument('--swav_features_weights_path', type=str, help='SwAV Features Weights Path', required=False)
    parser.add_argument('--no_dense_layer1_neurons', type=int, help='Dense Layer 1 inputs size', default=256)
    parser.add_argument('--no_dense_layer2_neurons', type=int, help='Dense Layer 2 inputs size', default=128)
    parser.add_argument('--no_dense_layer3_neurons', type=int, help='Dense Layer 3 inputs size', default=64)

    # Mirrored training arguments
    parser.add_argument('--mirrored_training', type=bool, help='Use Mirrored Training', default=False)
    parser.add_argument('--no_replicas', type=int, help='No Replicas', default=4)

    return parser.parse_args()


def get_model(args):
    from learning.model import models

    built_models = {
        OPTICAL_FLOW: lambda: models.get_opticalflow_model(),
        RGB: lambda: models.get_rgb_model(),
        NSDM: lambda: models.get_nsdm_model(),
        NSDMV2: lambda: models.get_nsdm_v2_model(),
        NSDMV3: lambda: models.get_nsdm_v3_model(lstm_cells=args.lstm_cells,
                                                 embedding_size=args.embedding_size,
                                                 load_weights=args.load_weights,
                                                 swav_features_weights_path=args.swav_features_weights_path,
                                                 no_dense_layer1_neurons=args.no_dense_layer1_neurons,
                                                 no_dense_layer2_neurons=args.no_dense_layer2_neurons,
                                                 no_dense_layer3_neurons=args.no_dense_layer3_neurons),
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


def get_dataset(model_name, batch_size, train_dataset_path, test_dataset_path=None):
    data_preparers = {
        OPTICAL_FLOW: lambda: OpticalflowDatasetPreparer(train_dataset_path, test_dataset_path=test_dataset_path),
        RGB: lambda: RGBDatasetPreparer(train_dataset_path, test_dataset_path=test_dataset_path),
        NSDM: lambda: CombinedDatasetPreparer(train_dataset_path, test_dataset_path=test_dataset_path),
        NSDMV2: lambda: CombinedDatasetPreparer(train_dataset_path, test_dataset_path=test_dataset_path),
        NSDMV3: lambda: RGBDatasetPreparer(train_dataset_path, test_dataset_path=test_dataset_path),
        SWAV: lambda: SerializedSwAVDatasetPreparer(train_dataset_path, test_dataset_path=test_dataset_path)
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

        NSDMV3: lambda: NSDMExecutorV3(start_learning_rate=args.start_learning_rate,
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

    # if args.model != SWAV:
    #     executor.configure(models_to_execute)

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


def get_distributed_train_step(distribute_strategy, train_step_fn, input_spec):
    @tf.function(input_signature=(input_spec,))
    def step(inputs):
        per_replica_losses = distribute_strategy.run(train_step_fn, args=(inputs,))
        # return distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return per_replica_losses

    return step


def get_distributed_optimizer(distribute_strategy, get_optimizer_fn, no_epochs, no_steps):
    with distribute_strategy.scope():
        optimizer = get_optimizer_fn(no_epochs, no_steps)
        return optimizer


def get_distributed_callback(distribute_strategy, get_callback_fn, checkpoint_storage_path, model):
    with distribute_strategy.scope():
        callback = get_callback_fn(checkpoint_storage_path, model)
        return callback


def configure_model(executor, models_to_execute):
    def configure():
        executor.configure(models_to_execute)
    return configure


def distributed_configure_model(distribute_strategy, configure_model_fn):
    with distribute_strategy.scope():
        configure_model_fn()


def get_distributed_compute_loss(distributed_strategy, get_loss_fn, global_batch_size):
    with distributed_strategy.scope():
        loss_object = get_loss_fn()

        @tf.function
        def compute_loss(true_val, pred_val, model_losses):
            per_example_loss = loss_object(true_val, pred_val)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

            return loss

        return compute_loss


def get_compute_loss(get_loss_fn):
    loss_object = get_loss_fn()

    @tf.function
    def compute_loss(true_val, pred_val, model_losses):
        per_example_loss = loss_object(true_val, pred_val)
        if model_losses:
            per_example_loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

        return per_example_loss

    return compute_loss


def main():
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'learning-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()

    distribute_strategy = get_distributed_strategy(args.no_replicas, logger) if args.mirrored_training else None
    global_batch_size = args.batch_size
    per_replica_batch_size = int(args.batch_size / distribute_strategy.num_replicas_in_sync) if args.mirrored_training else args.batch_size

    logger.debug('learning operation started with the following parameters: %s', args)

    get_model_fn = get_model(args)
    model = get_distributed_model(distribute_strategy, get_model_fn) if args.mirrored_training else get_model_fn()

    executor = get_executor(model, args)

    if args.model == SWAV:
        # Custom training loops require a distributed dataset to be passed
        get_dataset_fn = get_dataset(args.model,
                                     global_batch_size,
                                     args.train_dataset_path)

        dataset = get_distributed_dataset(distribute_strategy,
                                          get_dataset_fn) if args.mirrored_training else get_dataset_fn()

        optimizer = get_distributed_optimizer(distribute_strategy,
                                              executor.get_optimizer,
                                              args.no_epochs,
                                              args.no_steps) if args.mirrored_training else executor.get_optimizer(args.no_epochs,
                                                                                                                   args.no_steps)
        compute_loss_fn = get_distributed_compute_loss(distribute_strategy,
                                                       executor.get_distributed_loss,
                                                       global_batch_size=global_batch_size) if args.mirrored_training else get_compute_loss(executor.get_loss)

        # callback assignation
        callback = get_distributed_callback(distribute_strategy,
                                            executor.get_callback,
                                            args.checkpoint_storage_path,
                                            model) if args.mirrored_training else executor.get_callback(args.checkpoint_storage_path,
                                                                                                        model)

        # training step function assignation
        non_dist_train_step = executor.train_step(global_batch_size, per_replica_batch_size, compute_loss_fn)
        train_step_fn = get_distributed_train_step(distribute_strategy,
                                                   non_dist_train_step,
                                                   dataset.element_spec) if args.mirrored_training else non_dist_train_step

        executor.train_model(model,
                             dataset,
                             args.no_epochs,
                             args.no_steps,
                             model_storage_path=args.model_storage_path,
                             failure_storege_path=args.failure_reason_path)(train_step_fn, optimizer, callback)
    else:
        # Keras model.fit loops don't require a distributed dataset to be passed
        get_dataset_fn = get_dataset(args.model, global_batch_size, args.train_dataset_path)
        dataset = get_dataset_fn()
        configure_fn = configure_model(executor, [model])
        if args.mirrored_training:
            distributed_configure_model(distribute_strategy, configure_fn)
        else:
            configure_fn()

        executor.train_model([model], dataset, args.no_epochs)

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
