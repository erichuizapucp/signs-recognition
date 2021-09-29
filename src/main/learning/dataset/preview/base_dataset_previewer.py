import io
import tensorflow as tf
import matplotlib.pyplot as plt


class BaseDatasetPreviewer:
    def __init__(self, logs_folder, preparer):
        self.file_writer = tf.summary.create_file_writer(logs_folder)
        self.dataset: tf.data.Dataset = preparer.prepare_train_dataset()

    def preview_dataset(self):
        raise NotImplementedError('preview_dataset method not implemented.')

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
