import json
import sys
import time
import os
from watchdog.observers import Observer

from video_processing import VideoProcessing
from video_processing_watcher import VideoProcessingWatcher

if __name__ == '__main__':
    args = sys.argv[1:]
    incoming_dir = args[0]

    existing_files = [f for f in os.listdir(incoming_dir) if os.path.isfile(os.path.join(incoming_dir, f))]
    for file_name in existing_files:
        with open(os.path.join(incoming_dir, file_name)) as f:
            data = json.load(f)
            VideoProcessing.process(data)

    observer = Observer()
    observer.schedule(VideoProcessingWatcher(), path=incoming_dir if args else '.')
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
