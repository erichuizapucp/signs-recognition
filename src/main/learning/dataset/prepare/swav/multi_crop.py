import tensorflow as tf


class MultiCropTransformer:
    @tf.function
    def tie_together(self, video_fragment, min_scale, max_scale, crop_size):
        if video_fragment.dtype == tf.string:
            video_fragment = self.decode(video_fragment)

        video_fragment = self.scale(video_fragment)

        # Random resized crops
        video_fragment = self.random_resize_crop(video_fragment, min_scale, max_scale, crop_size)

        # Color distortions & Gaussian blur
        video_fragment = self.custom_augment(video_fragment)

        return video_fragment

    @staticmethod
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def random_resize_crop(video_fragment, min_scale, max_scale, crop_size):
        crop_resized_images = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        index = 0
        for image in video_fragment:
            seed = (index, 0)
            index = index + 1
            # Conditional resizing
            image_shape = 260 if crop_size == 224 else 160
            resized_image = tf.image.resize(image, (image_shape, image_shape))

            # Get the crop size for given min and max scale
            size = tf.random.uniform(shape=(1,), minval=min_scale * image_shape, maxval=max_scale * image_shape,
                                     dtype=tf.float32)
            size = tf.cast(size, tf.int32)[0]
            # Get the crop from the image
            crop = tf.image.stateless_random_crop(resized_image, size=[size, size, 3], seed=seed)
            crop_resize = tf.image.resize(crop, (crop_size, crop_size))
            crop_resized_images.append(crop_resize)

        return crop_resized_images.stack()

    @tf.function
    def custom_augment(self, video_fragment):
        video_fragment = self.random_apply(self.flip_left_right, video_fragment, p=0.5)

        # Randomly apply Gaussian blur
        video_fragment = self.random_apply(self.gaussian_blur, video_fragment, p=0.5)

        # Randomly apply transformation (color distortions) with probability p.
        video_fragment = self.random_apply(self.color_jitter, video_fragment, p=0.8)

        # Randomly apply grayscale
        video_fragment = self.random_apply(self.color_drop, video_fragment, p=0.1)

        return video_fragment

    @staticmethod
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def gaussian_blur(video_fragment, kernel_size=23, padding='SAME'):
        transformed_video_fragment = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

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

        return transformed_video_fragment.stack()

    @staticmethod
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def color_jitter(video_fragment, s=0.5):
        transformed_video_fragment = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        index = 0
        for image in video_fragment:
            seed = (index, 0)
            index = index + 1
            # random brightness is giving issues so it is removed from transformation for now.
            # transformed_image = tf.image.stateless_random_brightness(image, max_delta=0.8 * s, seed=seed)

            transformed_image = tf.image.stateless_random_contrast(image, lower=1-0.8 * s,
                                                                   upper=1 + 0.8 * s, seed=seed)
            transformed_image = tf.image.stateless_random_saturation(transformed_image, lower=1-0.8 * s,
                                                                     upper=1 + 0.8 * s, seed=seed)
            transformed_image = tf.image.stateless_random_hue(transformed_image, max_delta=0.2 * s, seed=seed)
            transformed_image = tf.clip_by_value(transformed_image, 0, 1)

            transformed_video_fragment.append(transformed_image)

        return transformed_video_fragment.stack()

    @staticmethod
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def color_drop(video_fragment):
        transformed_video_fragment = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for image in video_fragment:
            transformed_image = tf.image.rgb_to_grayscale(image)
            transformed_image = tf.tile(transformed_image, [1, 1, 3])

            transformed_video_fragment.append(transformed_image)

        return transformed_video_fragment.stack()

    @staticmethod
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def flip_left_right(video_fragment):
        transformed_video_fragment = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for image in video_fragment:
            transformed_image = tf.image.flip_left_right(image)
            transformed_video_fragment.append(transformed_image)

        return transformed_video_fragment.stack()

    @staticmethod
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def decode(video_fragment):
        decoded_video_fragment = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for image in video_fragment:
            # convert to integers
            decoded_image = tf.image.decode_jpeg(image, channels=3)
            decoded_video_fragment.append(decoded_image)

        return decoded_video_fragment.stack()

    @staticmethod
    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def scale(video_fragment):
        scaled_video_fragment = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for image in video_fragment:
            # convert to floats in the [0,1] range.
            image = tf.image.convert_image_dtype(image, tf.float32)
            scaled_video_fragment.append(image)

        return scaled_video_fragment.stack()

    @staticmethod
    def random_apply(transformation_func, video_fragment, p):
        return tf.cond(tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
                       lambda: transformation_func(video_fragment),
                       lambda: video_fragment)
