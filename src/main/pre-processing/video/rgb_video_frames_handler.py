import cv2

from common.handler import Handler


class RGBVideoFramesHandler(Handler):
    def __init__(self):
        print()

    def handle(self, **kwargs):
        video_path = kwargs['VideoPath']
        start_time = kwargs['StartTime']
        end_time = kwargs['EndTime']

        video_capture = cv2.VideoCapture(video_path)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        duration = float(frame_count) / float(fps)

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_time)

        frame_no = 0

        while True:
            success, frame = video_capture.read()
            if not success:  # eof
                break
            cv2.imwrite("frame%d.jpg" % frame_no, frame)
            frame_no += 1

        video_capture.release()
