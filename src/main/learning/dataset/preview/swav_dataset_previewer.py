import tensorflow as tf
import matplotlib.pyplot as plt

from learning.dataset.preview.base_dataset_previewer import BaseDatasetPreviewer
from datetime import datetime


class SwAVDatasetPreviewer(BaseDatasetPreviewer):
    def __init__(self, logs_folder, preparer):
        super().__init__(logs_folder, preparer)

    def preview_dataset(self):
        plot_images = []
        start_time = datetime.now()
        for multi_sample in self.dataset.take(5):
            figure = self._image_grid(multi_sample)
            plot_image = self._plot_to_image(figure)
            plot_images.append(plot_image)
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

        with self.file_writer.as_default():
            plot_images = tf.reshape(plot_images, [-1, 1000, 1000, 4])
            tf.summary.image("SwAV Training Data", plot_images, max_outputs=25, step=0)

    def _image_grid(self, multi_sample):
        figure = plt.figure(figsize=(10, 10))

        plot_index = 0
        for crop_index, multi_crop in enumerate(multi_sample):
            # preview only the first 5 frames per multi crop sample
            first_frames = multi_crop[:5]

            for frame_index, frame in enumerate(first_frames):
                plot_index += 1
                crop_w = frame.shape[0]
                crop_h = frame.shape[1]

                plot_title = self._get_image_label(crop_index=crop_index,
                                                   crop_w=crop_w,
                                                   crop_h=crop_h,
                                                   frame_index=frame_index)
                plt.subplot(5, 5, plot_index, title=plot_title)

                plt.xticks([])
                plt.yticks([])
                plt.grid(False)

                plt.imshow(tf.image.convert_image_dtype(frame, tf.int32))

        return figure

    def _get_image_label(self, **kwargs):
        crop_index = kwargs['crop_index'] + 1
        crop_w = kwargs['crop_w']
        crop_h = kwargs['crop_h']
        frame_index = kwargs['frame_index'] + 1

        return "TR: {0}-{1}x{2}-{3}".format(crop_index, crop_w, crop_h, frame_index)
