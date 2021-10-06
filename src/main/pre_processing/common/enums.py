from enum import Enum


class PreProcessingOperation(Enum):
    SAMPLES_GENERATION_SPLIT = 'samples-generation-split'
    AUDIO_TRANSCRIPTION = 'transcribe-audio'
    TRANSCRIPTION_SYNTAX_DETECTION = 'video-transcription-syntax-detection'
    TRANSCRIPTION_KEYWORDS_DETECTION = 'keywords-detection'
    ISOLATED_SAMPLES_METADATA_GENERATION = 'samples-metadata-generation'
    ISOLATED_RGB_SAMPLES_GENERATION = 'rgb-samples-generation'
    ISOLATED_OPTICALFLOW_SAMPLES_GENERATION = 'opticalflow-samples-generation'
