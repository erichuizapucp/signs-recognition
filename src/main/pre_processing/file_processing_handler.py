import logging

from common.enums import PreProcessingOperation as PreProcOp
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
            PreProcOp.SAMPLES_GENERATION_SPLIT: lambda x: VideoSplittingProcessor().process(x),
            PreProcOp.AUDIO_TRANSCRIPTION: lambda x: TranscriptionProcessor().process(x),
            PreProcOp.TRANSCRIPTION_SYNTAX_DETECTION: lambda x: TranscriptionSyntaxDetectionProcessor().process(x),
            PreProcOp.TRANSCRIPTION_KEYWORDS_DETECTION: lambda x: KeywordsDetectionProcessor().process(x),
            PreProcOp.ISOLATED_SAMPLES_METADATA_GENERATION: lambda x: SamplesMetadataGenerationProcessor().process(x),
            PreProcOp.ISOLATED_RGB_SAMPLES_GENERATION: lambda x: RGBSamplesProcessor().process(x),
            PreProcOp.ISOLATED_OPTICALFLOW_SAMPLES_GENERATION: lambda x: OpticalflowSamplesProcessor().process(x)
        }

        if operation in operations:
            self.logger.debug('An %s operation is started', operation)
            return operations[operation](data.get('payload'))
        else:
            raise ValueError('Unrecognized operation "{}"'.format(operation))
