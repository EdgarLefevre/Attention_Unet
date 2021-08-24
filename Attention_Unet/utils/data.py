import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


class Dataset(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        img_size,
        input_img_paths,
        label_img_paths
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.label_img_paths = label_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_label_img_paths = self.label_img_paths[i: i + self.batch_size]
        x = np.zeros(
            (self.batch_size, self.img_size, self.img_size, 1), dtype="float32"
        )
        for j, path in enumerate(batch_input_img_paths):
            img = (
                np.array(
                    keras.preprocessing.image.load_img(
                        path,
                        color_mode="grayscale",
                        target_size=(self.img_size, self.img_size),
                    )
                )
                / 255
            )
            x[j] = np.expand_dims(img, 2)

        y = np.zeros(
            (self.batch_size, self.img_size, self.img_size, 1), dtype="float32"
        )
        for k, path_lab in enumerate(batch_label_img_paths):
            img = (
                    np.array(
                        keras.preprocessing.image.load_img(
                            path_lab,
                            color_mode="grayscale",
                            target_size=(self.img_size, self.img_size),
                        )
                    )
                    / 255
            )
            y[k] = np.expand_dims(img, 2)
        return tf.constant(x), tf.constant(y)

