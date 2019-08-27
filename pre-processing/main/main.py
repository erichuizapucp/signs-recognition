import json
import time
import os
import io_utils
import logging
import logging.config
import yaml

from watchdog.observers import Observer
from video_processing import VideoProcessing
from video_processing_watcher import VideoProcessingWatcher


def setup_logging(working_folder, default_level=logging.INFO):
    logging_config = 'logging.yaml'

    logging_config = os.path.join(working_folder, logging_config)

    if os.path.exists(logging_config):
        with open(logging_config, 'r') as f:
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
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder)
    logger = logging.getLogger(__name__)

    incoming_queue = os.path.join(working_folder, io_utils.INCOMING_QUEUE)

    logger.info('STARTING THE VIDEO PRE-PROCESSING MODULE')

    existing_files = [f for f in os.listdir(incoming_queue) if os.path.isfile(os.path.join(incoming_queue, f))]
    for file_name in existing_files:
        with open(os.path.join(incoming_queue, file_name)) as f:
            data = json.load(f)
            VideoProcessing.process(data)

    observer = Observer()
    observer.schedule(VideoProcessingWatcher(), path=incoming_queue)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

    logger.info('ENDING THE VIDEO PRE-PROCESSING MODULE')


if __name__ == '__main__':
    main()
