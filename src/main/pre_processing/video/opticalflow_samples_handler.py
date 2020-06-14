import cv2 as cv
import os
import logging

from pre_processing.common.samples_handler import SamplesHandler
from pre_processing.video.opticalflow_samples_extractor import OpticalflowSamplesExtractor


class OpticalflowSamplesHandler(SamplesHandler):
    # params for ShiTomasi corner detection
    __feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    __lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def handle_sample(self, **kwargs):
        rgb_sample_folder_path = kwargs['RGBSampleFolderPath']
        opticalflow_sample_path = kwargs['OFSamplePath']

        frame_index = 1
        no_frames = len([name for name in os.listdir(rgb_sample_folder_path) if name != '.DS_Store'])
        frame_file_name = os.path.join(rgb_sample_folder_path, str(frame_index).zfill(4) + '_frame.jpg')

        frames = []
        while frame_index < no_frames:
            frame_index = frame_index + 1
            frame = cv.imread(frame_file_name)
            frames.append(frame)
            frame_file_name = os.path.join(rgb_sample_folder_path, str(frame_index).zfill(4) + '_frame.jpg')

        opticalflow_extractor = OpticalflowSamplesExtractor()
        opticalflow_img = opticalflow_extractor.extract_sample(RGBSample=frames)

        cv.imwrite(opticalflow_sample_path, opticalflow_img)
        self.logger.debug('Optical flow is available at: %s', opticalflow_sample_path)
