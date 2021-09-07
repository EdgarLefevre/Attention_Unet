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
import Attention_Unet.models.small_plus_plus as unetpp
import sklearn.model_selection as sk
import skimage.io as io
import skimage.transform as transform
import numpy as np
import progressbar
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]

# BASE_PATH = "/home/edgar/Documents/Datasets/JB/supervised/"
BASE_PATH = "/home/edgar/Documents/Datasets/Sim_cell_seg/"

def get_model(att):
    model_seg = unet.unet((512, 512, 1), filters=16, drop_r=0.5, attention=att)
    optim = tf.keras.optimizers.Adam(lr=0.0001)
    model_seg.compile(
        loss="bianry_crossentropy",
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
    dataset_val = data.Dataset(16, 512, img_val, label_val)
    return dataset_train, dataset_val


def create_pred_dataset(path_img):
    img = io.imread(path_img).astype(np.uint8)
    img = np.array(img) / 255
    img = data.contrast_and_reshape(img)
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
    base_path = BASE_PATH + "test/"
    # pathlist = [
    #     base_path + "Spheroid_D31000_02_w2soSPIM-405_135_5.png",
    #     base_path + "Spheroid_D31000_02_w2soSPIM-405_135_6.png",
    #     base_path + "Spheroid_D31000_02_w2soSPIM-405_135_9.png",
    #     base_path + "Spheroid_D31000_02_w2soSPIM-405_136_5.png",
    #     base_path + "Spheroid_D31000_02_w2soSPIM-405_136_9.png"
    # ]
    pathlist = [
        base_path + "0.png",
        base_path + "1.png",
        base_path + "2.png",
        base_path + "3.png",
        base_path + "4.png"
    ]
    imgs, preds = pred_(model, pathlist)
    utils.visualize(imgs, preds)


def _loss(model, image):
    loss_fn = tf.keras.metrics.binary_crossentropy
    pred = model(image)
    loss = tf.reduce_mean(loss_fn(image, pred))
    return loss


def _grad(model, image):
    with tf.GradientTape() as tape:
        loss = _loss(model, image)
    return loss, tape.gradient(loss, model.trainable_variables)


def _step(
        model,
        x,
        optimizer,
        train=True,
):
    loss, grads = _grad(model, x)
    if train:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def run_epoch(
        dataset,
        model,
        optim,
        train=True,
):
    loss_epoch = 0
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        for i, (x, w) in enumerate(dataset):
            bar.update(i)
            loss_step = _step(
                model,
                x,
                optim,
                train
            )
            loss_epoch += loss_step
    return loss_epoch


def _train(epochs, dataset, dataset_val, model, optimizer):
    loss_t = []
    loss_v = []
    for epoch in range(epochs):
        utils.print_gre("Epoch {}/{}:\n".format(epoch + 1, epochs))
        utils.print_gre("Training data:")
        loss_train = run_epoch(
            dataset,
            model,
            optimizer
        )
        utils.print_gre("Validation data:")
        loss_val = run_epoch(
            dataset_val,
            model,
            optimizer,
            train=False
        )
        utils.print_gre(
            "\nEpoch {} : \n\tTraining loss : {}\n\tValidation loss : {}\n".format(
                epoch + 1, np.array(loss_train).mean(), np.array(loss_val).mean()
            )
        )
        loss_t.append(loss_train)
        loss_v.append(loss_val)
    utils.learning_curves(loss_t, loss_v)


def train(path_images, path_labels):
    model = unet.unet((512, 512, 1), filters=8, drop_r=0.2, attention=True)
    dataset_train, dataset_val = get_datasets(path_images, path_labels)
    _train(50,
           dataset_train,
           dataset_val,
           model=model,
           optimizer=tf.keras.optimizers.Adam(lr=0.002)
           )
    pred(model)
    viz_att_map(model)


def plot_att_map(img, map):
    fig = plt.figure(figsize=(15, 10))
    columns = 2
    rows = 1  # nb images
    ax = []
    ax.append(fig.add_subplot(rows, columns, 1))
    ax[-1].set_title("Input")
    plt.imshow(img, cmap="gray")
    ax.append(fig.add_subplot(rows, columns, 2))
    ax[-1].set_title("Attention map")
    plt.imshow(map * 255)
    plt.colorbar()
    # plt.show()
    fig.savefig("plots/att_map.png")
    plt.close(fig)



def viz_att_map(model):
    # image = np.array(
    #     io.imread(BASE_PATH + "test/Spheroid_D31000_02_w2soSPIM-405_135_6.png")) / 255
    image = np.array(
        io.imread(BASE_PATH + "test/2.png")) / 255
    # image = data.contrast_and_reshape(image)
    image = image.reshape(1, 512, 512, 1)
    output = model.get_layer("block7_att").output
    intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=output)
    map = intermediate_model(image)
    plot_att_map(image.reshape(512, 512), map.numpy().reshape(256, 256))


if __name__ == "__main__":
    print("Tensorflow version : ", tf.__version__)
    train(
        path_images=BASE_PATH + "imgs/",
        path_labels=BASE_PATH + "labels/",
    )
