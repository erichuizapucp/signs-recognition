import logging
import os

from learning.model.opticalflow_model_builder import OpticalFlowModelBuilder
from learning.model.rgb_recurrent_model_builder import RGBRecurrentModelBuilder
from learning.model.novel_signs_detection_builder import NovelSignsDetectionModelBuilder
from argparse import ArgumentParser
from logger_config import setup_logging
from learning.execution.opticalflow_executor import OpticalflowExecutor
from learning.execution.rgb_executor import RGBExecutor
from learning.execution.nsdm_executor import NsdmExecutor
from learning.common.model_type import OPTICAL_FLOW, RGB, NSDM

DEFAULT_NO_EPOCHS = 5
DEFAULT_NO_STEPS_EPOCHS = None


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', help='Model Name', required=True)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=DEFAULT_NO_EPOCHS)
    parser.add_argument('-ns', '--no_steps', help='Number of steps per epoch', default=DEFAULT_NO_STEPS_EPOCHS)

    return parser.parse_args()


def get_saved_model(model_name):
    models = {
        OPTICAL_FLOW: lambda: OpticalFlowModelBuilder(),
        RGB: lambda: RGBRecurrentModelBuilder(),
    }
    model = models[model_name]().load_saved_model()
    return model


def get_model(model_name):
    models = {
        OPTICAL_FLOW: lambda: OpticalFlowModelBuilder(),
        RGB: lambda: RGBRecurrentModelBuilder(),
        NSDM: lambda: NovelSignsDetectionModelBuilder(get_saved_model(OPTICAL_FLOW), get_saved_model(RGB)),
    }
    model = models[model_name]().build()
    return model


def get_executor(executor_name, model):
    executors = {
        OPTICAL_FLOW: lambda: OpticalflowExecutor(model),
        RGB: lambda: RGBExecutor(model),
        NSDM: lambda: NsdmExecutor(model)
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
    no_epochs = args.no_epochs
    no_steps_per_epoch = args.no_steps
    model_name = args.model
    executor_name = model_name

    logger.debug('learning operation started with the following parameters: %s', args)

    model = get_model(model_name)
    executor = get_executor(executor_name, model)
    executor.train_model(no_epochs, no_steps_per_epoch)

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
