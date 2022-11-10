import cv2
import tensorflow as tf
from pre_processing.common.samples_extractor import SamplesExtractor


def add_extracted_frame(extracted_frames, frame_no, end_frame_no, video_capture):
    is_frame_captured, frame = video_capture.read()
    extracted_frames = tf.cond(tf.constant(is_frame_captured),
                               lambda: extracted_frames.write(frame_no, frame), lambda: extracted_frames)

    return [extracted_frames, tf.math.add(frame_no, 1), end_frame_no, video_capture]


def get_all_extracted_frames(extracted_frames, frame_no, end_frame_no, video_capture):
    extracted_frames, _, _, _ = tf.while_loop(add_extracted_frame_cond, add_extracted_frame,
                                              [extracted_frames, frame_no, end_frame_no, video_capture])
    return extracted_frames


def add_extracted_frame_cond(extracted_frames, frame_no, end_frame_no, video_capture):
    return frame_no < end_frame_no


def add_extracted_person_box_cond(extracted_frames, bounding_box, frame_no, end_frame_no, video_capture):
    return frame_no < end_frame_no


def get_video_capture(video_path, start_time, end_time):
    video_capture = cv2.VideoCapture(str(video_path))
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # frames per second

    start_frame_no = int(start_time * fps)
    end_frame_no = int(end_time * fps)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)

    return video_capture, start_frame_no, end_frame_no


class RGBSamplesExtractor(SamplesExtractor):
    def __init__(self, detect_person=False, detection_model=None):
        self.detect_person = tf.constant(detect_person)
        self.detection_model = detection_model

    @tf.function(input_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                  tf.TensorSpec(shape=(), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.float32)))
    def extract_sample(self, video_path, start_time, end_time):
        # tf.print(video_path)
        # print(video_path)

        # video_capture, start_frame_no, end_frame_no = tf.numpy_function(get_video_capture, [video_path, start_time, end_time], [cv2.VideoCapture, tf.int32, tf.int32])
        # video_capture = cv2.VideoCapture(tf.compat.path_to_str(video_path).replace('tf.Tensor(b\'', '').replace('\', shape=(), dtype=string)', ''))
        # video_capture = cv2.VideoCapture(tf.compat.path_to_str(video_path))
        video_capture = cv2.VideoCapture(video_path)
        fps = 30  # frames per second

        start_frame_no = tf.cast(tf.math.multiply(start_time, fps), dtype=tf.int32)
        end_frame_no = tf.cast(tf.math.multiply(end_time, fps), dtype=tf.int32)

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_no = start_frame_no
        extracted_frames = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)

        while tf.less(frame_no, end_frame_no):
            is_frame_captured, frame = video_capture.read()
            extracted_frames = tf.cond(tf.constant(is_frame_captured),
                                       lambda: extracted_frames.write(frame_no, frame), lambda: extracted_frames)

            frame_no = tf.math.add(frame_no, 1)

        # is_frame_captured, frame = video_capture.read()

        # could not detect the first frame, video is in corrupt state
        # if not is_frame_captured:
        #     return False, None

        # extracted_frames = tf.cond(self.detect_person,
        #                            lambda: self.add_frames_for_detected_person(extracted_frames, frame_no, end_frame_no,
        #                                                                        video_capture),
        #                            lambda: get_all_extracted_frames(extracted_frames, frame_no, end_frame_no,
        #                                                             video_capture))

        # if self.detect_person and is_frame_captured:
        #     is_bounding_box_detected, bounding_box = self.get_detection_bounding_box(frame)
        #     if is_bounding_box_detected:
        #         # return False, self.empty_sample_person
        #         while frame_no < end_frame_no and is_frame_captured:
        #             person_crop = self.extract_person_from_frame(frame, bounding_box)
        #             extracted_frames = extracted_frames.write(frame_no, person_crop)
        #
        #             frame_no += 1
        #             is_frame_captured, frame = video_capture.read()
        #     else:
        #         return False, None
        # elif not self.detect_person and is_frame_captured:
        #     extracted_frames, _, _, _, _, _ = \
        #         tf.while_loop(add_extract_frame_cond,
        #                       add_extracted_frame,
        #                       [extracted_frames, frame_no, end_frame_no, frame, is_frame_captured, video_capture])
        #
        #     # while frame_no < end_frame_no and is_frame_captured:
        #     #     _, _extracted_frames = extracted_frames.write(frame_no, frame)
        #     #
        #     #     frame_no += 1
        #     #     is_frame_captured, frame = video_capture.read()
        # else:
        #     return False, None

        video_capture.release()

        return extracted_frames.stack()

    def add_frames_for_detected_person(self, extracted_frames, frame_no, end_frame_no, video_capture):
        is_frame_captured, frame = video_capture.read()
        is_bounding_box_detected, bounding_box = tf.cond(tf.constant(is_frame_captured),
                                                         lambda: self.get_detection_bounding_box(frame),
                                                         lambda: (False, tf.zeros(4)))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return tf.cond(tf.constant(is_bounding_box_detected),
                       lambda: self.get_all_person_frames(extracted_frames, bounding_box, frame_no,
                                                          end_frame_no, video_capture),
                       lambda: extracted_frames)

    def get_all_person_frames(self, extracted_frames, bounding_box, frame_no, end_frame_no, video_capture):
        extracted_frames, _, _, _, _ = tf.while_loop(add_extracted_person_box_cond, self.add_extracted_person_frame,
                                                     [extracted_frames, bounding_box, frame_no, end_frame_no, video_capture])

        return extracted_frames

    def add_extracted_person_frame(self, extracted_frames, bounding_box, frame_no, end_frame_no, video_capture):
        is_frame_captured, frame = video_capture.read()
        is_person_captured, person_crop = tf.cond(tf.constant(is_frame_captured),
                                                  lambda: (True, self.extract_person_from_frame(frame, bounding_box)),
                                                  lambda: (False, tf.zeros(shape=[672, 448, 3])))
        extracted_frames = tf.cond(tf.constant(is_person_captured),
                                   lambda: extracted_frames.write(frame_no, person_crop),
                                   lambda: extracted_frames)

        return [extracted_frames, bounding_box, tf.math.add(frame_no, 1), end_frame_no, video_capture]

    # @tf.function(input_signature=(tf.TensorSpec([1280, 720, 3], dtype=tf.int32),))
    def get_detection_bounding_box(self, frame):
        label_id_offset = 1
        input_tensor = tf.expand_dims(frame, axis=0)

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

    # @tf.function(input_signature=(tf.TensorSpec([1, 1280, 720, 3], dtype=tf.int32),))
    def detect_person_in_frame(self, frame):
        image, shapes = self.detection_model.preprocess(frame)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)

        return detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']

    # @tf.function(input_signature=(tf.TensorSpec([1280, 720, 3], dtype=tf.int32), tf.TensorSpec(4, dtype=tf.float32)))
    def extract_person_from_frame(self, image, detection_box):
        frame_width = 1280
        frame_height = 720

        output_width = 672  # standard image width (ensuring 3X 224 crops horizontally)
        output_height = 448  # standard image height (ensuring 2X 224 crops vertically)

        y_min = tf.cast(tf.math.multiply(tf.gather(detection_box, 0), frame_height), dtype=tf.int32)
        x_min = tf.cast(tf.math.multiply(tf.gather(detection_box, 1), frame_width), dtype=tf.int32)
        y_max = tf.cast(tf.math.multiply(tf.gather(detection_box, 2), frame_height), dtype=tf.int32)
        x_max = tf.cast(tf.math.multiply(tf.gather(detection_box, 3), frame_width), dtype=tf.int32)

        cropped_img = tf.image.crop_to_bounding_box(image, y_min, x_min, tf.math.subtract(y_max, y_min),
                                                    tf.math.subtract(x_max, x_min))

        # aspect ratio is not preserved because the generator requires a 4 dimensional tensor and not a list of tensors
        cropped_img = tf.image.resize(cropped_img, [output_width, output_height], preserve_aspect_ratio=False)

        return cropped_img
