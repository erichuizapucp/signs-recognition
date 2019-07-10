import sys
import time
from watchdog.observers import Observer

from video_processing_watcher import VideoProcessingWatcher

if __name__ == '__main__':
    args = sys.argv[1:]
    observer = Observer()
    observer.schedule(VideoProcessingWatcher(), path=args[0] if args else '.')
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
