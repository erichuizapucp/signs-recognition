import io
import tensorflow as tf
import matplotlib.pyplot as plt


class BaseDatasetPreviewer:
    def __init__(self, logs_folder, preparer):
        self.file_writer = tf.summary.create_file_writer(logs_folder)
        self.dataset: tf.data.Dataset = preparer.prepare_train_dataset()

    def preview_dataset(self):
        plot_images = []
        for multi_sample in self.dataset.take(20):
            figure = self._image_grid(multi_sample)
            plot_image = self._plot_to_image(figure)
            plot_images.append(plot_image)

        with self.file_writer.as_default():
            plot_images = tf.reshape(plot_images, [-1, 1000, 1000, 4])
            tf.summary.image("SwAV Training Data", plot_images, max_outputs=25, step=0)

    @staticmethod
    def _plot_to_image(figure):
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        # close figure to save resources
        plt.close(figure)
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def _image_grid(self, multi_sample):
        raise NotImplementedError('image_grid method not implemented.')

    def _get_image_label(self, **kwargs):
        raise NotImplementedError('_get_image_label method not implemented.')
