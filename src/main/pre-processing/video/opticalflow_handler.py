import cv2 as cv
import numpy as np
import os
import logging

from common import io_utils
from common.handler import Handler


class OpticalFlowHandler(Handler):
    # params for ShiTomasi corner detection
    __feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    __lk_params = dict(winSize=(15, 15),maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def handle(self, **kwargs):
        rgb_sample_folder_path = kwargs['RGBSampleFolderPath']
        opticalflow_sample_path = kwargs['OFSamplePath']

        # Create some random colors
        # color = np.random.randint(0, 255, (100, 3))
        color = [255, 0, 128]

        frame_index = 1
        no_frames = len([name for name in os.listdir(rgb_sample_folder_path)])
        frame_file_name = os.path.join(rgb_sample_folder_path, 'frame' + str(frame_index) + '.jpg')

        # Take first frame and find corners in it
        old_frame = cv.imread(frame_file_name)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.__feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        while frame_index < no_frames:
            frame_index = frame_index + 1
            frame_file_name = os.path.join(rgb_sample_folder_path, 'frame' + str(frame_index) + '.jpg')

            frame = cv.imread(frame_file_name)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.__lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                # mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                # frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
                mask = cv.line(mask, (a, b), (c, d), color, 2)
                frame = cv.circle(frame, (a, b), 5, color, -1)
            img = cv.add(frame, mask)
            cv.imwrite(opticalflow_sample_path, img)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        self.logger.debug('Optical flow is available at: %s', opticalflow_sample_path)
