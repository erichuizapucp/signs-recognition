import tensorflow as tf


class MultiCropTransformer:
    def tie_together(self, video_fragment, min_scale, max_scale, crop_size):
        video_fragment = self.decode(video_fragment)
        video_fragment = self.scale(video_fragment)

        # Random resized crops
        video_fragment = self.random_resize_crop(video_fragment, min_scale, max_scale, crop_size)

        # Color distortions & Gaussian blur
        video_fragment = self.custom_augment(video_fragment)

        return video_fragment

    @staticmethod
    def random_resize_crop(video_fragment, min_scale, max_scale, crop_size):
        crop_resized_images = []

        for image in video_fragment:
            # Conditional resizing
            if crop_size == 224:
                image_shape = 260
                image = tf.image.resize(image, (image_shape, image_shape))
            else:
                image_shape = 160
                image = tf.image.resize(image, (image_shape, image_shape))

            # Get the crop size for given min and max scale
            size = tf.random.uniform(shape=(1,), minval=min_scale * image_shape, maxval=max_scale * image_shape,
                                     dtype=tf.float32)
            size = tf.cast(size, tf.int32)[0]
            # Get the crop from the image
            crop = tf.image.random_crop(image, (size, size, 3))
            crop_resize = tf.image.resize(crop, (crop_size, crop_size))
            crop_resized_images.append(crop_resize)

        return crop_resized_images

    def custom_augment(self, video_fragment):
        video_fragment = self.random_apply(tf.image.flip_left_right, video_fragment, p=0.5)

        # Randomly apply Gaussian blur
        video_fragment = self.random_apply(self.gaussian_blur, video_fragment, p=0.5)

        # Randomly apply transformation (color distortions) with probability p.
        video_fragment = self.random_apply(self.color_jitter, video_fragment, p=0.8)

        # Randomly apply grayscale
        video_fragment = self.random_apply(self.color_drop, video_fragment, p=0.2)

        return video_fragment

    @staticmethod
    def gaussian_blur(video_fragment, kernel_size=23, padding='SAME'):
        transformed_video_fragment = []

        for image in video_fragment:
            sigma = tf.random.uniform((1,)) * 1.9 + 0.1

            radius = tf.cast(kernel_size / 2, tf.int32)
            kernel_size = radius * 2 + 1
            x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
            blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
            blur_filter /= tf.reduce_sum(blur_filter)

            # One vertical and one horizontal filter.
            blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
            blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
            num_channels = tf.shape(image)[-1]
            blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
            blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
            expand_batch_dim = image.shape.ndims == 3

            if expand_batch_dim:
                image = tf.expand_dims(image, axis=0)
            blurred = tf.nn.depthwise_conv2d(image, blur_h, strides=[1, 1, 1, 1], padding=padding)
            blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)

            if expand_batch_dim:
                blurred = tf.squeeze(blurred, axis=0)

            transformed_video_fragment.append(blurred)

        return transformed_video_fragment

    @staticmethod
    def color_jitter(video_fragment, s=0.5):
        transformed_video_fragment = []
        for image in video_fragment:
            transformed_image = tf.image.random_brightness(image, max_delta=0.8 * s)
            transformed_image = tf.image.random_contrast(transformed_image, lower=1-0.8 * s, upper=1 + 0.8 * s)
            transformed_image = tf.image.random_saturation(transformed_image, lower=1-0.8 * s, upper=1 + 0.8 * s)
            transformed_image = tf.image.random_hue(transformed_image, max_delta=0.2 * s)
            transformed_image = tf.clip_by_value(transformed_image, 0, 1)

            transformed_video_fragment.append(transformed_image)

        return transformed_video_fragment

    @staticmethod
    def color_drop(video_fragment):
        transformed_video_fragment = []
        for image in video_fragment:
            transformed_image = tf.image.rgb_to_grayscale(image)
            transformed_image = tf.tile(transformed_image, [1, 1, 3])

            transformed_video_fragment.append(transformed_image)

        return transformed_video_fragment

    @staticmethod
    def flip_left_right(video_fragment):
        transformed_video_fragment = []
        for image in video_fragment:
            transformed_image = tf.image.flip_left_right(image)
            transformed_video_fragment.append(transformed_image)

        return transformed_video_fragment

    @staticmethod
    def decode(video_fragment):
        transformed_video_fragment = []
        for image in video_fragment:
            # convert to integers
            decoded_image = tf.image.decode_jpeg(image, channels=3)
            transformed_video_fragment.append(decoded_image)

        return transformed_video_fragment

    @staticmethod
    def scale(video_fragment):
        transformed_video_fragment = []
        for image in video_fragment:
            # convert to floats in the [0,1] range.
            image = tf.image.convert_image_dtype(image, tf.float32)
            transformed_video_fragment.append(image)

        return transformed_video_fragment

    @staticmethod
    def random_apply(transformation_func, video_fragment, p):
        return tf.cond(tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
                       lambda: transformation_func(video_fragment),
                       lambda: video_fragment)
