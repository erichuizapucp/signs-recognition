import logging

from audio.audio_processor import AudioProcessor
from text_analysis.keywords_detection_processor import KeywordsDetectionProcessor
from text_analysis.transcription_syntax_detection_processor import TranscriptionSyntaxDetectionProcessor
from transcription.transcription_processor import TranscriptionProcessor
from video.video_splitting_processor import VideoSplittingProcessor
from text_analysis.samples_metadata_generation_processor import SamplesMetadataGenerationProcessor
from video.rgb_samples_processor import RGBSamplesProcessor
from video.opticalflow_samples_processor import OpticalflowSamplesProcessor


class FileProcessingHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def handle(self, data):
        operation = data['operation']

        operations = {
            'extract-audio': lambda x: AudioProcessor().process(x),
            'video-split': lambda x: VideoSplittingProcessor().process(x),
            'transcribe-audio': lambda x: TranscriptionProcessor().process(x),
            'transcription-syntax-detection': lambda x: TranscriptionSyntaxDetectionProcessor().process(x),
            'keywords-detection': lambda x: KeywordsDetectionProcessor().process(x),
            'samples-metadata-generation': lambda x: SamplesMetadataGenerationProcessor().process(x),
            'rgb-samples-generation': lambda x: RGBSamplesProcessor().process(x),
            'opticalflow-samples-generation': lambda x: OpticalflowSamplesProcessor().process(x)
        }

        if operation in operations:
            self.logger.debug('An %s operation is started', operation)
            return operations[operation](data.get('payload'))
        else:
            raise ValueError('Unrecognized operation "{}"'.format(operation))
