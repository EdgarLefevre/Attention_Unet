# -*- coding: utf-8 -*-
# !/usr/bin/python3.8
# -*- coding: utf-8 -*-

"""
Train
==========
This file is used to train networks to segment images.
"""

import os

import tensorflow as tf
import Attention_Unet.utils.utils as utils
import Attention_Unet.utils.data as data
import Attention_Unet.models.model_unet as unet
import sklearn.model_selection as sk

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_model(att):
    model_seg = unet.unet((128, 128, 1), filters=8, drop_r=0.2, attention=att)
    optim = tf.keras.optimizers.Adam(lr=0.001)
    loss_fn = "binary_crossentropy"
    model_seg.compile(
        loss=loss_fn,
        optimizer=optim,
    )
    return model_seg


def get_datasets(path_img, path_label):
    img_path_list = utils.list_files_path(path_img)
    label_path_list = utils.list_files_path(path_label)
    # not good if we need to do metrics
    img_train, img_val, label_train, label_val = sk.train_test_split(
        img_path_list, label_path_list, test_size=0.2, random_state=42
    )

    dataset_train = data.Dataset(16, 512, img_train, label_train)
    dataset_val = data.Dataset(2, 512, img_val, label_val)
    return dataset_train, dataset_val


def train(path_images, path_labels):
    # earlystopper = keras.callbacks.EarlyStopping(
    #     monitor="val_" + metric,
    #     patience=args["patience"],
    #     verbose=1,
    #     min_delta=0.001,
    #     restore_best_weights=True,
    #     mode="min",
    # )
    # cb_list = [earlystopper, utils.CosLRDecay(args["n_epochs"], args["lr"]), checkpoint]

    model_seg = get_model(True)
    dataset_train, dataset_val = get_datasets(path_images, path_labels)
    print(dataset_train[0])  # issue here, check labels value
    history = model_seg.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=25,
        # callbacks=cb_list,
    )
    utils.plot_learning_curves(
        history, "segmentation_attention_unet", "loss"
    )


if __name__ == "__main__":
    train(
        path_images="/home/edgar/Documents/Datasets/JB/supervised/imgs/",
        path_labels="/home/edgar/Documents/Datasets/JB/supervised/labels/",
    )
