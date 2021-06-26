import cv2
from pre_processing.common.samples_extractor import SamplesExtractor


class RGBSamplesExtractor(SamplesExtractor):
    def extract_sample(self, **kwargs):
        video_path = kwargs['VideoPath']
        start_time = float(kwargs['StartTime'])
        end_time = float(kwargs['EndTime'])

        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # frames per second

        start_frame_no = 0 if int(start_time * fps) - 1 < 0 else int(start_time * fps) - 1
        end_frame_no = 0 if int(end_time * fps) - 1 < 0 else int(end_time * fps) - 1

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

        frame_no = start_frame_no
        img_index = 1
        extracted_frames = []
        while frame_no <= end_frame_no:
            success, frame = video_capture.read()
            if not success:  # eof
                break
            frame_no += 1
            img_index += 1

            extracted_frames.append(frame)

        video_capture.release()

        return extracted_frames
