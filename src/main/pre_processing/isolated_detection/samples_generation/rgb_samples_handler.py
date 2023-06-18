import os
import cv2
import logging


from pre_processing.common import io_utils
from pre_processing.common.samples_handler import SamplesHandler
from rgb_person_sample_extractor import RGBPersonSamplesExtractor
from learning.common.object_detection_utility import ObjectDetectionUtility


class RGBSamplesHandler(SamplesHandler):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        object_detection_utility = ObjectDetectionUtility()
        person_detection_model_name = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
        person_detection_checkout_prefix = 'ckpt-0'
        detection_model = object_detection_utility.get_object_detection_model(person_detection_model_name,
                                                                              person_detection_checkout_prefix)
        self.extractor = RGBPersonSamplesExtractor(detection_model)

    def handle_sample(self, **kwargs):
        video_path = kwargs['VideoPath']
        start_time = float(kwargs['StartTime'])
        end_time = float(kwargs['EndTime'])
        token = kwargs['TokenName']
        dataset_path = kwargs['DataSetPath']

        success, tf_frames_array = self.extractor.extract_sample(video_path, start_time, end_time)
        if success:
            frames = tf_frames_array.stack()

            sample_folder_path = self.get_sample_folder_path(token, dataset_path)
            io_utils.check_path_dir(sample_folder_path)

            for frame_index, frame in enumerate(frames):
                frame = frame.numpy()
                frame = cv2.cvtColor(frame, cv2.cv2.COLOR_RGB2BGR)
                frame_file_path = os.path.join(sample_folder_path, str(frame_index + 1).zfill(4) + '_frame.jpg')
                cv2.imwrite(frame_file_path, frame)

    def get_sample_folder_path(self, token, dataset_path):
        token_folder_path = os.path.join(self.work_dir, dataset_path, token)
        io_utils.check_path_dir(token_folder_path)

        num_dirs = len([folder for folder in os.listdir(token_folder_path) if os.path.isdir(
            os.path.join(token_folder_path, folder))])

        return os.path.join(token_folder_path, str(num_dirs + 1))
