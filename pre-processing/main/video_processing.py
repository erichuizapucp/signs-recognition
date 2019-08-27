from audio.audio_processor import AudioProcessor
from transcription.transcription_processor import TranscriptionProcessor
from video.video_splitter import VideoSplitter


class VideoProcessing:
    @staticmethod
    def process(data):
        operation = data['operation']

        operations = {
            'extract-audio': lambda x: AudioProcessor().process(x),
            'video-split': lambda x: VideoSplitter().process(x),
            'transcribe-audio': lambda x: TranscriptionProcessor().process(x)
        }

        if operation in operations:
            return operations[operation](data.get('payload'))
        else:
            raise ValueError('Unrecognized operation "{}"'.format(operation))