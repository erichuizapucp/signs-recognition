import json
import os
from watchdog.events import PatternMatchingEventHandler
from video_processing import VideoProcessing


class VideoProcessingWatcher(PatternMatchingEventHandler):
    patterns = ["*.json"]

    @staticmethod
    def on_watch_event(event):
        if event.event_type == 'created' or event.event_type == 'modified':
            with open(event.src_path) as f:
                data = json.load(f)
                VideoProcessing.process(data)

            # delete the file when the process is completed
            os.remove(event.src_path)

    def on_modified(self, event):
        self.on_watch_event(event)

    def on_created(self, event):
        self.on_watch_event(event)
