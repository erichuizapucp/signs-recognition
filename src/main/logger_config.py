import os
import logging
import logging.config
import yaml


def setup_logging(working_folder, config_file_name, default_level=logging.INFO):
    logging_config = os.path.join(working_folder, config_file_name)

    if os.path.exists(logging_config):
        with open(logging_config, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())

                core_log_file_path = os.path.join(working_folder, config['handlers']['core_file_handler']['filename'])
                config['handlers']['core_file_handler']['filename'] = core_log_file_path

                if 'training_file_handler' in config['handlers']:
                    training_log_file_path = os.path.join(working_folder,
                                                          config['handlers']['training_file_handler']['filename'])
                    config['handlers']['training_file_handler']['filename'] = training_log_file_path

                logging.config.dictConfig(config)
            except Exception as e:
                print(e)
                print('Error in logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print('Failed to load configuration file. Using default configs')
