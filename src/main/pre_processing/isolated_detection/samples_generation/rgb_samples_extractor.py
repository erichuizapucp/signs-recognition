import cv2
from pre_processing.common.samples_extractor import SamplesExtractor


class RGBSamplesExtractor(SamplesExtractor):
    def extract_sample(self, video_path, start_time, end_time):
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # frames per second

        start_frame_no = int(start_time * fps)
        end_frame_no = int(end_time * fps)

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

        frame_no = start_frame_no
        extracted_frames = []

        while frame_no < end_frame_no:
            is_frame_captured, frame = video_capture.read()
            frame_no += 1

            extracted_frames = extracted_frames.extend([frame])

        video_capture.release()

        return extracted_frames
