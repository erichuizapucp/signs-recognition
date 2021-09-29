import cv2
import numpy as np
import tensorflow as tf
from pre_processing.common.samples_extractor import SamplesExtractor


class RGBSamplesExtractor(SamplesExtractor):
    def extract_sample(self, **kwargs):
        video_path = kwargs['VideoPath']
        start_time = float(kwargs['StartTime'])
        end_time = float(kwargs['EndTime'])
        detect_person = kwargs['DetectPerson']
        detection_model = kwargs['DetectionModel']

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

            if detect_person:
                is_detected, person_crop = self.extract_person_from_frame(frame, detection_model)
                if is_detected:
                    extracted_frames.append(person_crop)
            else:
                extracted_frames.append(frame)

        video_capture.release()

        return extracted_frames

    def extract_person_from_frame(self, frame, detection_model):
        label_id_offset = 1
        input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = self.__detect_person(input_tensor, detection_model)

        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
        detection_scores = detections['detection_scores'][0].numpy()

        if detection_classes[0] == 1 and detection_scores[0] > 0.5:
            cropped_image = self.__crop_person_bounding_box(frame, detection_boxes[0])
            return True, cropped_image
        return False, frame

    @staticmethod
    def __detect_person(image, model):
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    @staticmethod
    def __crop_person_bounding_box(image, detection_box):
        frame_height, frame_width = image.shape[:2]

        standard_width = 672  # standard image width (ensuring 3X 224 crops horizontally)
        standard_height = 448  # standard image height (ensuring 2X 224 crops vertically)

        y_min = int(detection_box[0] * frame_height)
        x_min = int(detection_box[1] * frame_width)
        y_max = int(detection_box[2] * frame_height)
        x_max = int(detection_box[3] * frame_width)
        cropped_img = tf.image.crop_to_bounding_box(image, y_min, x_min, y_max - y_min, x_max - x_min)

        # aspect ratio is not preserved because the generator requires a 4 dimensional tensor and not a list of tensors
        cropped_img = tf.image.resize(cropped_img, [standard_width, standard_height], preserve_aspect_ratio=False)

        return cropped_img
