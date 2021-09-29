import random

import cv2
import tensorflow as tf

from learning.dataset.prepare.raw_dataset_preparer import RawDatasetPreparer
from pre_processing.isolated_detection.samples_generation.rgb_samples_extractor import RGBSamplesExtractor
from learning.dataset.prepare.swav.multi_crop import MultiCropTransformer


class SwAVDatasetPreparer(RawDatasetPreparer):
    def __init__(self,
                 train_dataset_path,
                 test_dataset_path,
                 object_detection_model):
        super().__init__(train_dataset_path, test_dataset_path)

        self.crop_sizes = [224, 96]
        self.num_crops = [2, 3]
        self.min_scale = [0.5, 0.14]
        self.max_scale = [1., 0.5]

        self.multi_crop = MultiCropTransformer()

        self.object_detection_model = object_detection_model

    def _prepare(self, dataset_path):
        return super()._prepare(dataset_path)

    def _data_generator(self, list_video_path):
        extractor = RGBSamplesExtractor()

        for video_path in list_video_path:
            str_video_path = tf.compat.as_str_any(video_path)

            duration = self.__get_video_duration(str_video_path)
            start_time = 0.0
            end_time = self.__get_next_end_time(start_time=start_time)

            while end_time < duration:
                fragment_frames = extractor.extract_sample(VideoPath=str_video_path,
                                                           StartTime=start_time,
                                                           EndTime=end_time,
                                                           DetectPerson=True,
                                                           DetectionModel=self.object_detection_model)

                start_time = end_time
                end_time = self.__get_next_end_time(start_time=start_time)

                if len(fragment_frames) > 0:
                    yield fragment_frames

    @staticmethod
    def __get_next_end_time(start_time):
        fragment_duration = random.uniform(0.3, 0.5)
        end_time = start_time + fragment_duration
        return end_time

    def _prepare_sample3(self, video_fragment):
        crops = tuple()
        for idx, num_crop in enumerate(self.num_crops):
            for _ in range(num_crop):
                transformed_video_fragment = self.multi_crop.tie_together(video_fragment,
                                                                          self.min_scale[idx],
                                                                          self.max_scale[idx],
                                                                          self.crop_sizes[idx])
                crops += (transformed_video_fragment,)
        return crops

    @staticmethod
    def __get_video_duration(video_path):
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        video_capture.release()

        return duration

    def _get_raw_file_types(self):
        return ['*.mp4', '*.avi']

    def _prepare_sample(self, feature, label):
        raise NotImplementedError('This method is not supported for SwAVDatasetPreparer.')

    def _prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('This method is not supported for SwAVDatasetPreparer.')
