import logging
import os
import cv2 as cv
import tensorflow as tf

from pre_processing.video.opticalflow_samples_extractor import OpticalflowSamplesExtractor
from pre_processing.video.rgb_samples_extractor import RGBSamplesExtractor

from legacy.nsdm_v2_builder import NSDMV2ModelBuilder
from learning.dataset.prepare.combined_dataset_preparer import CombinedDatasetPreparer

from learning.common.labels import SIGNS_CLASSES


class VideoEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.working_dir = os.getenv('WORK_DIR', './')

        self.start_time = 0.0
        self.extract_length = 0.5  # 0.5 seconds of extracted data
        self.offset = 0.1  # move 0.1 after each extraction
        self.min_threshold = 0.5

        self.data_preparer = CombinedDatasetPreparer()

    def evaluate(self, video_path):
        model_builder = NSDMV2ModelBuilder()
        model = model_builder.load_saved_model()

        video_duration = self.__get_video_duration(video_path)

        rgb_samples_extractor = RGBSamplesExtractor()
        opticalflow_samples_extractor = OpticalflowSamplesExtractor()

        start_time = self.start_time
        end_time = start_time + self.extract_length
        # extract samples for 1 second video and move 0.5 seconds through the end of the video.
        while end_time < video_duration:
            rgb_sample = rgb_samples_extractor.extract_sample(VideoPath=video_path, StartTime=start_time,
                                                              EndTime=end_time)
            opticalflow_sample = opticalflow_samples_extractor.extract_sample(RGBSample=rgb_sample)

            transformed_input = self.data_preparer.transform_feature_for_predict(OpticalflowFeature=opticalflow_sample,
                                                                                 RGBFeature=rgb_sample)

            # tensor with classes probabilities
            probabilities = model(transformed_input, training=False)
            # get a boolean 2D tensor determining if a class probability is greater than minimal prob threshold
            match_condition = tf.greater_equal(probabilities, tf.constant(self.min_threshold))
            # get indexes for probabilities greater than minimal prob threshold
            match_indices = tf.where(match_condition)

            # check if there are probabilities greater than minimal prob threshold
            if tf.not_equal(tf.size(match_indices), 0):
                predicted_index = tf.argmax(probabilities, 1)
                label = tf.gather(SIGNS_CLASSES, predicted_index)

                print("The sign {} was found at start_time: {} and end_time: {}".format(label, start_time, end_time))

            start_time = start_time + self.offset
            end_time = end_time + self.offset

    @staticmethod
    def __get_video_duration(video_path):
        video_capture = cv.VideoCapture(video_path)

        fps = video_capture.get(cv.CAP_PROP_FPS)
        frame_count = video_capture.get(cv.CAP_PROP_FRAME_COUNT)
        duration = float(frame_count) / float(fps)

        return duration
