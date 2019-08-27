import json
import time
import os
import io_utils
import logging_utils

from watchdog.observers import Observer

from video_processing import VideoProcessing
from video_processing_watcher import VideoProcessingWatcher

if __name__ == '__main__':
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    logging_utils.setup_logging(working_folder)
    logger = logging_utils.get_logger(logging_utils.PRE_PROCESSING_LOGGER)

    incoming_queue = os.path.join(working_folder, io_utils.INCOMING_QUEUE)

    # logger.info('STARTING THE VIDEO PRE-PROCESSING MODULE')

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

    # logger.info('ENDING THE VIDEO PRE-PROCESSING MODULE')
