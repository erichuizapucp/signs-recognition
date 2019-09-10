import json
import time
import os
import io_utils
import logging
import logging.config
import yaml

from watchdog.observers import Observer
from incoming_queue_watcher import IncomingQueueWatcher
from file_processing_handler import FileProcessingHandler


def setup_logging(default_level=logging.INFO):
    logging_config = 'logging.yaml'

    logging_config = os.path.join(working_folder, logging_config)

    if os.path.exists(logging_config):
        with open(logging_config, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(e)
                print('Error in logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print('Failed to load configuration file. Using default configs')


def main():
    incoming_queue = os.path.join(working_folder, io_utils.incoming_queue_folder)

    logger.info('Pre-processing is initiated with the following parameters')
    logger.info('Incoming Queue: %s', incoming_queue)
    logger.info('Working Folder: %s', working_folder)

    existing_files = [f for f in os.listdir(incoming_queue) if os.path.isfile(os.path.join(incoming_queue, f))]
    for file_name in existing_files:
        logger.debug('processing ''%s''', file_name)
        with open(os.path.join(incoming_queue, file_name)) as f:
            data = json.load(f)
            FileProcessingHandler().handle(data)

    observer = Observer()
    observer.schedule(IncomingQueueWatcher(), path=incoming_queue)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == '__main__':
    # obtain working folder from WORK_DIR environment variable
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # execute video pre-processing logic
    main()
