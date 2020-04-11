import logging
import os

from learning.model.opticalflow_model_builder import OpticalFlowModelBuilder
from learning.model.rgb_recurrent_model_builder import RGBRecurrentModelBuilder
from learning.model.novel_signs_detection_builder import NovelSignsDetectionModel
from argparse import ArgumentParser
from logger_config import setup_logging
from learning.execution.opticalflow_executor import OpticalflowExecutor
from learning.execution.rgb_executor import RGBExecutor
from learning.execution.nsdm_executor import NsdmExecutor

DEFAULT_NO_EPOCHS = 5
DEFAULT_NO_STEPS_EPOCHS = None


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', help='Model Name', required=True)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=DEFAULT_NO_EPOCHS)
    parser.add_argument('-ns', '--no_steps', help='Number of steps per epoch', default=DEFAULT_NO_STEPS_EPOCHS)

    return parser.parse_args()


def get_model(model_name):
    models = {
        'opticalflow': lambda: OpticalFlowModelBuilder(),
        'rgb': lambda: RGBRecurrentModelBuilder(),
        'nsdm': lambda: NovelSignsDetectionModel(),
    }
    model = models[model_name]().build()
    return model


def get_executor(executor_name, model, working_folder):
    executors = {
        'opticalflow': lambda: OpticalflowExecutor(model, working_folder),
        'rgb': lambda: RGBExecutor(model, working_folder),
        'nsdm': lambda: NsdmExecutor(model, working_folder)
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
    executor = get_executor(executor_name, model, working_folder)
    executor.train_model(no_epochs, no_steps_per_epoch)

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
