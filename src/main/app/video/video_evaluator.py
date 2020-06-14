import logging
import os
import cv2

from pre_processing.video.opticalflow_samples_extractor import OpticalflowSamplesExtractor
from pre_processing.video.rgb_samples_extractor import RGBSamplesExtractor

from learning.model.nsdm_builder import NSDMModelBuilder
from learning.execution.nsdm_executor import NSDMExecutor


class VideoEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.working_dir = os.getenv('WORK_DIR', './')

    def evaluate(self, video_path):
        model = NSDMModelBuilder().load_saved_model()

        video_duration = self.__get_video_duration(video_path)

        rgb_samples_extractor = RGBSamplesExtractor()
        opticalflow_samples_extractor = OpticalflowSamplesExtractor()

        start_time = 0.0
        end_time = 1.0
        # extract samples for 1 second video and move 0.5 seconds through the end of the video.
        while end_time < video_duration:
            start_time = start_time + 0.5
            end_time = end_time + 0.5

            rgb_sample = rgb_samples_extractor.extract_sample(VideoPath=video_path, StartTime=start_time,
                                                              EndTime=end_time)
            opticalflow = opticalflow_samples_extractor.extract_sample(RGBSample=rgb_sample)



    @staticmethod
    def __get_video_duration(video_path):
        video_capture = cv2.VideoCapture(video_path)

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = float(frame_count) / float(fps)

        return duration
