import cv2
import tensorflow as tf
from pre_processing.common.samples_extractor import SamplesExtractor


class RGBPersonSamplesExtractor(SamplesExtractor):
    def __init__(self, detection_model):
        self.detection_model = detection_model

    def extract_sample(self, video_path, start_time, end_time):
        video_capture = cv2.VideoCapture(video_path)
        fps = round(video_capture.get(cv2.CAP_PROP_FPS), 0)  # frames per second

        start_frame_no = int(start_time * fps)
        end_frame_no = int(end_time * fps)

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

        frame_no = start_frame_no
        extracted_frames = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True, clear_after_read=True)

        is_frame_captured, frame = video_capture.read()
        if not is_frame_captured:
            return False, None

        frame = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB)
        is_bounding_box_detected, bounding_box = self.get_detection_bounding_box(frame)
        if not is_bounding_box_detected:
            return False, None

        while frame_no < end_frame_no and is_frame_captured:
            is_person_detected, person_crop = self.extract_person_from_frame(frame, bounding_box)
            if is_person_detected:
                person_crop = tf.cast(person_crop, dtype=tf.uint8)
                extracted_frames = extracted_frames.write(frame_no, person_crop)

            frame_no += 1
            is_frame_captured, frame = video_capture.read()
            if is_frame_captured:
                frame = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB)

        video_capture.release()

        if extracted_frames.size() == 0:
            return False, None

        return True, extracted_frames

    def get_detection_bounding_box(self, frame):
        label_id_offset = 1
        input_tensor = tf.cast(tf.expand_dims(frame, axis=0), dtype=tf.float32)

        detection_boxes, detection_classes, detection_scores = self.detect_person_in_frame(input_tensor)

        boxes = tf.gather(detection_boxes, 0)
        classes = tf.math.add(tf.gather(detection_classes, 0), label_id_offset)
        scores = tf.gather(detection_scores, 0)

        detected_class = tf.gather(classes, 0)
        detected_score = tf.gather(scores, 0)

        res = tf.cond(tf.logical_and(tf.equal(detected_class, 1), tf.greater(detected_score, 0.5)),
                      lambda: (True, tf.gather(boxes, 0)),
                      lambda: (False, tf.zeros(4)))

        return res

    def detect_person_in_frame(self, frame):
        image, shapes = self.detection_model.preprocess(frame)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)

        return detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']

    def extract_person_from_frame(self, image, detection_box):
        if not image.shape == (720, 1280, 3):
            return False, None

        frame_width = 1280
        frame_height = 720

        output_width = 448  # standard image width (ensuring 3X 224 crops horizontally)
        output_height = 672  # standard image height (ensuring 2X 224 crops vertically)

        y_min = tf.cast(tf.math.multiply(tf.gather(detection_box, 0), frame_height), dtype=tf.int32)
        x_min = tf.cast(tf.math.multiply(tf.gather(detection_box, 1), frame_width), dtype=tf.int32)
        y_max = tf.cast(tf.math.multiply(tf.gather(detection_box, 2), frame_height), dtype=tf.int32)
        x_max = tf.cast(tf.math.multiply(tf.gather(detection_box, 3), frame_width), dtype=tf.int32)

        cropped_img = tf.image.crop_to_bounding_box(image, y_min, x_min, tf.math.subtract(y_max, y_min),
                                                    tf.math.subtract(x_max, x_min))

        # aspect ratio is not preserved because the generator requires a 4 dimensional tensor and not a list of tensors
        cropped_img = tf.image.resize(cropped_img, [output_height, output_width], preserve_aspect_ratio=False)

        return True, cropped_img
