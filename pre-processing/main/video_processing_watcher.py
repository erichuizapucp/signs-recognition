import json
import os
from watchdog.events import PatternMatchingEventHandler
from audio.audio_processor import AudioProcessor
from video.video_splitter import VideoSplitter


class VideoProcessingWatcher(PatternMatchingEventHandler):
    patterns = ["*.json"]

    @staticmethod
    def __process_videos(event):
        operation = event['operation']

        operations = {
            'extract-audio': lambda x: AudioProcessor().process(x),
            'video-split': lambda x: VideoSplitter().process(x)
        }

        if operation in operations:
            return operations[operation](event.get('payload'))
        else:
            raise ValueError('Unrecognized operation "{}"'.format(operation))

    def process(self, event):
        if event.event_type == 'created' or event.event_type == 'modified':
            with open(event.src_path) as f:
                data = json.load(f)
                self.__process_videos(data)

            # delete the file when the process is completed
            os.remove(event.src_path)

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)
