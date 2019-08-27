import logging
import logging.config
import yaml
import coloredlogs
import os

PRE_PROCESSING_LOGGER = 'pre_processing_logger'
LOGGING_CONFIG = 'logging.yaml'


def setup_logging(working_folder, default_level=logging.INFO):
    logging_config = os.path.join(working_folder, LOGGING_CONFIG)

    if os.path.exists(logging_config):
        with open(logging_config, 'r') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                coloredlogs.install()
            except Exception as e:
                print(e)
                print('Error in logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')


def get_logger(logger_name):
    logging.getLogger(logger_name)
