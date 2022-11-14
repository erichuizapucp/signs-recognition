import cv2
import os

from pre_processing.common.samples_handler import SamplesHandler
from rgb_samples_extractor import RGBSamplesExtractor


class RGBSamplesHandler(SamplesHandler):
    def handle_sample(self, **kwargs):
        video_path = kwargs['VideoPath']
        start_time = float(kwargs['StartTime'])
        end_time = float(kwargs['EndTime'])
        folder_path = kwargs['FolderPath']

        extractor = RGBSamplesExtractor()
        frames = extractor.extract_sample(video_path, start_time, end_time)

        for frame_index, frame in enumerate(frames):
            frame_file_path = os.path.join(folder_path, str(frame_index).zfill(4) + '_frame.jpg')
            cv2.imwrite(frame_file_path, frame)
