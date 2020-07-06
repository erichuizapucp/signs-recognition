import logging
import os

from learning.model.opticalflow_model_builder import OpticalFlowModelBuilder
from learning.model.rgb_recurrent_model_builder import RGBRecurrentModelBuilder
from learning.model.nsdm_builder import NSDMModelBuilder
from learning.model.nsdm_v2_builder import NSDMV2ModelBuilder
from argparse import ArgumentParser
from logger_config import setup_logging
from learning.execution.opticalflow_executor import OpticalflowExecutor
from learning.execution.rgb_executor import RGBExecutor
from learning.execution.nsdm_executor import NSDMExecutor
from learning.execution.nsdm_v2_executor import NSDMExecutorV2
from learning.common.model_type import OPTICAL_FLOW, RGB, NSDM, NSDMV2

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


def get_model(model_name):
    models = {
        OPTICAL_FLOW: lambda: get_opticalflow_model(),
        RGB: lambda: get_rgb_model(),
        NSDM: lambda: get_nsdm_model(),
        NSDMV2: lambda: get_nsdm_v2_model(),
    }
    model = models[model_name]()
    return model


def get_executor(executor_name, model):
    executors = {
        OPTICAL_FLOW: lambda: OpticalflowExecutor(model),
        RGB: lambda: RGBExecutor(model),
        NSDM: lambda: NSDMExecutor(model),
        NSDMV2: lambda: NSDMExecutorV2(model),
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
    executor_name = model_name

    logger.debug('learning operation started with the following parameters: %s', args)

    model = get_model(model_name)
    executor = get_executor(executor_name, model)
    executor.train_model(no_epochs, no_steps_per_epoch)

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
