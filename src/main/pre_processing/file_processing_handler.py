import logging

from common import processing_ops
from isolated_detection.text_analysis.keywords_detection_processor import KeywordsDetectionProcessor
from isolated_detection.text_analysis.transcription_syntax_detection_processor \
    import TranscriptionSyntaxDetectionProcessor
from isolated_detection.video_transcription.transcription_processor import TranscriptionProcessor
from isolated_detection.samples_generation.video_splitting_processor import VideoSplittingProcessor
from isolated_detection.text_analysis.samples_metadata_generation_processor import SamplesMetadataGenerationProcessor
from isolated_detection.samples_generation.rgb_samples_processor import RGBSamplesProcessor
from isolated_detection.samples_generation.opticalflow_samples_processor import OpticalflowSamplesProcessor


class FileProcessingHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def handle(self, data):
        operation = data['operation']

        operations = {
            processing_ops.SAMPLES_GENERATION_SPLIT:
                lambda payload: VideoSplittingProcessor().process(payload),
            processing_ops.AUDIO_TRANSCRIPTION:
                lambda payload: TranscriptionProcessor().process(payload),
            processing_ops.TRANSCRIPTION_SYNTAX_DETECTION:
                lambda payload: TranscriptionSyntaxDetectionProcessor().process(payload),
            processing_ops.TRANSCRIPTION_KEYWORDS_DETECTION:
                lambda payload: KeywordsDetectionProcessor().process(payload),
            processing_ops.ISOLATED_SAMPLES_METADATA_GENERATION:
                lambda payload: SamplesMetadataGenerationProcessor().process(payload),
            processing_ops.ISOLATED_RGB_SAMPLES_GENERATION:
                lambda payload: RGBSamplesProcessor().process(payload),
            processing_ops.ISOLATED_OPTICALFLOW_SAMPLES_GENERATION:
                lambda payload: OpticalflowSamplesProcessor().process(payload)
        }

        if operation in operations:
            self.logger.debug('An %s operation is started', operation)
            return operations[operation](data.get('payload'))
        else:
            raise ValueError('Unrecognized operation "{}"'.format(operation))
