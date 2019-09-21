import cv2

from common.handler import Handler


class OpticalFlowHandler(Handler):
    def handle(self, **kwargs):
        cv2.calcOpticalFlowPyrLK()
