import random

import tensorflow as tf

from learning.dataset.prepare.raw_dataset_preparer import RawDatasetPreparer
from pre_processing.isolated_detection.samples_generation.rgb_person_sample_extractor import RGBPersonSamplesExtractor
from learning.dataset.prepare.swav import multi_crop
from learning.common import video_utility


class SwAVDatasetPreparer(RawDatasetPreparer):
    def __init__(self,
                 train_dataset_path,
                 test_dataset_path,
                 person_detection_model,
                 **kwargs):
        super().__init__(train_dataset_path, test_dataset_path)

        self.crop_sizes: list[int] = kwargs['crop_sizes']
        self.num_crops: list[int] = kwargs['num_crops']
        self.min_scale: list[float] = kwargs['min_scale']
        self.max_scale: list[float] = kwargs['max_scale']

        self.extractor = RGBPersonSamplesExtractor(person_detection_model)

        self.random_hash = {}

    def prepare_dataset(self, dataset_path, batch_size):
        return super().prepare_dataset(dataset_path, batch_size)

    def data_generator(self, list_video_path):  # sequentially read video samples
        for video_path in list_video_path:
            str_video_path = tf.compat.as_str_any(video_path)

            duration = video_utility.get_video_duration(str_video_path)
            start_time = 0.0
            end_time = self.__get_next_end_time(start_time=start_time)

            while end_time < duration:
                fragment_frames = self.extractor.extract_sample(str_video_path, start_time, end_time)

                start_time = end_time
                end_time = self.__get_next_end_time(start_time=start_time)

                if len(fragment_frames) > 0:
                    yield fragment_frames

    def data_generator2(self, video_path_list, chunk_start_list, chunk_end_list):  # randomly read video samples
        for index, video_path in enumerate(video_path_list):
            str_video_path = tf.compat.as_str_any(video_path)
            chunk_start = chunk_start_list[index]
            chunk_end = chunk_end_list[index]

            start_time = chunk_start
            end_time = self.__get_next_end_time(start_time=start_time)

            while end_time < chunk_end:
                success, fragment_frames = self.extractor.extract_sample(str_video_path, start_time, end_time)

                start_time = end_time
                end_time = self.__get_next_end_time(start_time=start_time)

                if success:
                    yield fragment_frames

    @staticmethod
    def __get_next_end_time(start_time):
        fragment_duration = random.uniform(0.3, 0.5)
        end_time = start_time + fragment_duration
        return end_time

    def prepare_sample3(self, video_fragment):
        crops = tuple()
        for idx, num_crop in enumerate(self.num_crops):
            for _ in range(num_crop):
                transformed_video_fragment = multi_crop.tie_together(video_fragment,
                                                                     self.min_scale[idx],
                                                                     self.max_scale[idx],
                                                                     self.crop_sizes[idx])
                crops += (transformed_video_fragment,)

        return crops

    def get_raw_file_types(self):
        return ['*.mp4', '*.avi']

    def prepare_sample(self, feature, label):
        raise NotImplementedError('This method is not supported for SwAVDatasetPreparer.')

    def prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('This method is not supported for SwAVDatasetPreparer.')
