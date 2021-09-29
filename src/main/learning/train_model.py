import os
import logging
import tensorflow as tf

from argparse import ArgumentParser
from logger_config import setup_logging

from learning.model.legacy.opticalflow_model_builder import OpticalFlowModelBuilder
from learning.model.legacy.rgb_recurrent_model_builder import RGBRecurrentModelBuilder
from learning.model.legacy.nsdm_builder import NSDMModelBuilder
from learning.model.legacy.nsdm_v2_builder import NSDMV2ModelBuilder
from learning.model.swav.swav_builder import SwAVModelBuilder

from learning.execution.legacy.opticalflow_executor import OpticalflowExecutor
from learning.execution.legacy.rgb_executor import RGBExecutor
from learning.execution.legacy.nsdm_executor import NSDMExecutor
from learning.execution.legacy.nsdm_v2_executor import NSDMExecutorV2
from learning.execution.swav.swav_executor import SwAVExecutor

from learning.common.model_type import OPTICAL_FLOW, RGB, NSDM, NSDMV2, SWAV

from learning.common.model_utility import ModelUtility

DEFAULT_NO_EPOCHS = 5
DEFAULT_NO_STEPS_EPOCHS = None


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', help='Model Name', required=True)
    parser.add_argument('-tr', '--train_dataset_path', help='Train Dataset Path', required=True)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=DEFAULT_NO_EPOCHS)
    parser.add_argument('-ns', '--no_steps', help='Number of steps per epoch', default=DEFAULT_NO_STEPS_EPOCHS)
    parser.add_argument('-odm', '--object_detection_model_name', help='Object Detection Model Name', required=False)
    parser.add_argument('-odcp', '--object_detection_checkout_prefix', help='Object Detection Checkout Prefix',
                        required=False)

    return parser.parse_args()


def get_saved_model(model_name):
    models = {
        OPTICAL_FLOW: lambda: OpticalFlowModelBuilder(),
        RGB: lambda: RGBRecurrentModelBuilder(),
        SWAV: lambda: SwAVModelBuilder()
    }
    model = models[model_name]().load_saved_model()
    return model


def get_opticalflow_model():
    model_builder = OpticalFlowModelBuilder()
    opticalflow_model = model_builder.build()
    return opticalflow_model


def get_rgb_model():
    model_builder = RGBRecurrentModelBuilder()
    rgb_model = model_builder.build()
    return rgb_model


def get_nsdm_model():
    opticalflow_model = get_saved_model(OPTICAL_FLOW)
    rgb_model = get_saved_model(RGB)

    model_builder = NSDMModelBuilder()
    nsdm_model = model_builder.build(OpticalflowModel=opticalflow_model, RGBModel=rgb_model)
    return nsdm_model


def get_nsdm_v2_model():
    opticalflow_model = get_saved_model(OPTICAL_FLOW)
    rgb_model = get_saved_model(RGB)

    model_builder = NSDMV2ModelBuilder()
    nsdm_v2_model = model_builder.build(OpticalflowModel=opticalflow_model, RGBModel=rgb_model)
    return nsdm_v2_model


def get_swav_models():
    model_builder = SwAVModelBuilder()
    feature_embeddings_model = model_builder.build()
    prototype_projections_model = model_builder.build2()

    return feature_embeddings_model, prototype_projections_model


def get_model(model_name):
    models = {
        OPTICAL_FLOW: lambda: get_opticalflow_model(),
        RGB: lambda: get_rgb_model(),
        NSDM: lambda: get_nsdm_model(),
        NSDMV2: lambda: get_nsdm_v2_model(),
        SWAV: lambda: get_swav_models()
    }
    model = models[model_name]()
    return model


def get_executor(executor_name, model, train_dataset_path, **kwargs):
    executors = {
        OPTICAL_FLOW: lambda: OpticalflowExecutor(model=model, train_dataset_path=train_dataset_path),
        RGB: lambda: RGBExecutor(model=model, train_dataset_path=train_dataset_path),
        NSDM: lambda: NSDMExecutor(model=model, train_dataset_path=train_dataset_path),
        NSDMV2: lambda: NSDMExecutorV2(model=model, train_dataset_path=train_dataset_path),
        SWAV: lambda: SwAVExecutor(feature_detection_model=model[0],
                                   projection_model=model[1],
                                   train_dataset_path=train_dataset_path,
                                   object_detection_model=kwargs['ObjectDetectionModel'])
    }
    executor = executors[executor_name]()
    executor.configure()
    return executor


def main():
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'learning-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()
    no_epochs = int(args.no_epochs)
    no_steps_per_epoch = args.no_steps
    model_name = args.model
    train_dataset_path = args.train_dataset_path
    executor_name = model_name
    object_detection_model_name = args.object_detection_model_name
    object_detection_checkout_prefix = args.object_detection_checkout_prefix

    logger.debug('learning operation started with the following parameters: %s', args)

    model = get_model(model_name)

    model_utility = ModelUtility()
    object_detection_model = model_utility.get_object_detection_model(object_detection_model_name,
                                                                      object_detection_checkout_prefix)
    executor = get_executor(executor_name, model, train_dataset_path)
    executor.train_model(no_epochs, no_steps_per_epoch)

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
