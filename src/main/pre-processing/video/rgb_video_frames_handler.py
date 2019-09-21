import cv2
import os

from common.handler import Handler


class RGBVideoFramesHandler(Handler):
    def handle(self, **kwargs):
        video_path = kwargs['VideoPath']
        start_time = kwargs['StartTime']
        end_time = kwargs['EndTime']
        folder_path = kwargs['FolderPath']

        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        # frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # duration = float(frame_count) / float(fps)

        start_frame_no = 0 if int(start_time * fps) - 1 < 0 else int(start_time * fps) - 1
        end_frame_no = 0 if int(end_time * fps) - 1 < 0 else int(end_time * fps) - 1

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

        frame_no = start_frame_no
        while frame_no <= end_frame_no:
            frame_file_path = os.path.join(folder_path, "frame%d.jpg" % frame_no)
            success, frame = video_capture.read()
            if not success:  # eof
                break
            cv2.imwrite(frame_file_path, frame)
            frame_no += 1

        video_capture.release()
