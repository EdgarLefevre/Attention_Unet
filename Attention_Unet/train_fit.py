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
import skimage.io as io
import skimage.transform as transform
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_model(att):
    model_seg = unet.unet((512, 512, 1), filters=8, drop_r=0.5, attention=att)
    optim = tf.keras.optimizers.Adam(lr=0.0001)
    loss_fn = "binary_crossentropy"
    model_seg.compile(
        loss=loss_fn,
        optimizer=optim,
    )
    return model_seg


def get_datasets(path_img, path_label):
    img_path_list = utils.list_files_path(path_img)
    label_path_list = utils.list_files_path(path_label)
    img_path_list, label_path_list = utils.shuffle_lists(img_path_list, label_path_list)

    # not good if we need to do metrics
    img_train, img_val, label_train, label_val = sk.train_test_split(
        img_path_list, label_path_list, test_size=0.2, random_state=42
    )

    dataset_train = data.Dataset(16, 512, img_train, label_train)
    dataset_val = data.Dataset(2, 512, img_val, label_val)
    return dataset_train, dataset_val


def create_pred_dataset(path_img):
    img = io.imread(path_img).astype(np.uint8)
    img = np.array(img) / 255
    img = np.array(img).reshape(-1, 512, 512, 1)
    return img


def pred_(model, path_list):
    pred_list = []
    img_list = []
    for path in path_list:
        img = create_pred_dataset(path)
        res = model.predict(img)
        img_list.append(img.reshape(512, 512) * 255)
        pred_list.append((res > 0.5).astype(np.uint8).reshape(512, 512) * 255)
    return img_list, pred_list


def pred(model):
    base_path = "/home/edgar/Documents/Datasets/JB/supervised/test/"
    pathlist = [
        base_path + "Spheroid_D31000_02_w2soSPIM-405_135_5.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_135_6.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_135_9.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_136_5.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_136_9.png"
    ]
    imgs, preds = pred_(model, pathlist)
    utils.visualize(imgs, preds)


def train(path_images, path_labels):
    model_seg = get_model(True)
    dataset_train, dataset_val = get_datasets(path_images, path_labels)
    history = model_seg.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=200,
    )
    utils.plot_learning_curves(
        history, "segmentation_attention_unet", "loss"
    )
    pred(model_seg)


if __name__ == "__main__":
    train(
        path_images="/home/edgar/Documents/Datasets/JB/supervised/imgs/",
        path_labels="/home/edgar/Documents/Datasets/JB/supervised/labels/",
    )
