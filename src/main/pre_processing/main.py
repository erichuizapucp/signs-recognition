import json
import time
import os
import io_utils
import logging.config

from watchdog.observers import Observer
from incoming_queue_watcher import IncomingQueueWatcher
from file_processing_handler import FileProcessingHandler

from logger_config import setup_logging


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
    setup_logging(working_folder, 'pre-processing-logging.yaml')
    logger = logging.getLogger(__name__)

    # execute samples_generation pre_processing logic
    main()
