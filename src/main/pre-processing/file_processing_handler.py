import logging

from audio.audio_processor import AudioProcessor
from transcription.transcription_processor import TranscriptionProcessor
from video.video_splitter import VideoSplitter


class FileProcessingHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def handle(self, data):
        operation = data['operation']

        operations = {
            'extract-audio': lambda x: AudioProcessor().process(x),
            'video-split': lambda x: VideoSplitter().process(x),
            'transcribe-audio': lambda x: TranscriptionProcessor().process(x)
        }

        if operation in operations:
            self.logger.debug('An %s operation is started', operation)
            return operations[operation](data.get('payload'))
        else:
            raise ValueError('Unrecognized operation "{}"'.format(operation))
