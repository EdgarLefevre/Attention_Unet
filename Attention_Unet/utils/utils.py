#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
import random

import matplotlib.pyplot as plt
import skimage.measure as measure
import tensorflow as tf
import tensorflow.keras.backend as K


def list_files_path(path):
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return sorted_alphanumeric([path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def plot_learning_curves(history, name, metric, path="plots/"):
    """
    Plot training curves.

    :param history: Result of model.fit
    :param name: Name of the plot (saving name)
    :type name: str
    :param metric: Metric to monitor
    :type metric: str
    :param path: Saving path
    :type path: str

    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history[metric]
    val_acc = history.history["val_" + metric]
    epochs = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if metric != "loss":
        ax1.plot(epochs, acc, label="Entraînement")
        ax1.plot(epochs, val_acc, label="Validation")
        ax1.set_title("Précision - Données entraînement vs. validation.")
        ax1.set_ylabel("Précision (" + metric + ")")
        ax1.set_xlabel("Epoch")
        ax1.legend()

    ax2.plot(epochs, loss, label="Entraînement")
    ax2.plot(epochs, val_loss, label="Validation")
    ax2.set_title("Perte - Données entraînement vs. validation.")
    ax2.set_ylabel("Perte")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.savefig(path + name + ".png")
    plt.close(fig)


def shuffle_lists(lista, listb, seed=42):
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def print_red(skk):
    """
    Print in red.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    """
    Print in green.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[92m{}\033[00m".format(skk))


def sorted_alphanumeric(data):
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


class LRDecay(tf.keras.callbacks.Callback):
    """
    Callback used to linearly reduce the learning rate.

    :param epoch_decay: Number of epoch to do before reduce learning rate
    :type epoch_decay: int
    :param coef: Reduce coefficient
    :type coef: int
    """

    def __init__(self, epoch_decay, coef=10):
        super().__init__()
        self.epoch_decay = epoch_decay
        self.coef = coef

    def on_epoch_begin(self, epoch, logs=None):
        """
        At the beginning of each epoch, if enough epochs are done, reduce the learning rate by coef.
        """
        if (epoch + 1) % self.epoch_decay == 0:
            self.model.optimizer.lr = self.model.optimizer.lr / self.coef
            print_gre("\nLearning rate is {}".format(self.model.optimizer.lr.numpy()))


class CosLRDecay(tf.keras.callbacks.Callback):
    """
    Callback used to perform cosine learning rate decay.

    .. note::
       Idea come from : https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx

    :param nb_epochs: Number total of epoch to run.
    :type nb_epochs: int
    """

    def __init__(self, nb_epochs, lr):
        super().__init__()
        # self.f_lr = self.model.optimizer.lr
        self.nb_epochs = nb_epochs

    def on_epoch_begin(self, epoch, logs=None):
        """
        At the beginning of each epoch, process a new learning rate.
        """
        self.model.optimizer.lr = (
                0.5
                * (1 + math.cos(epoch * math.pi / self.nb_epochs))
                * self.model.optimizer.lr
        )
        if self.model.optimizer.lr == 0.0:
            self.model.optimizer.lr = 1e-10
        print_gre("\nLearning rate is {}".format(self.model.optimizer.lr.numpy()))


def visualize(imgs, pred):
    fig = plt.figure(figsize=(15, 10))
    columns = 2
    rows = 5  # nb images
    ax = []  # loop around here to plot more images
    i = 0
    for j, img in enumerate(imgs):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Input")
        plt.imshow(img, cmap="gray")
        ax.append(fig.add_subplot(rows, columns, i + 2))
        ax[-1].set_title("Mask")
        plt.imshow(pred[j], cmap="gray")
        i += 2
        if i >= 15:
            break
    # plt.show()
    fig.savefig("plots/prediction.png")
    plt.close(fig)


def learning_curves(train, val):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.suptitle("Training Curves")
    ax.plot(train, label="Entraînement")
    ax.plot(val, label="Validation")
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=14)
    fig.savefig("plots/plot.png")
    plt.close(fig)


def get_args():
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: Dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=50, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=16, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--attention", "-a", dest="att", action="store_true", help="If flag, use attention block"
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Size of the image, one number"
    )
    parser.add_argument(
        "--drop_r", "-d", type=float, default=0.2, help="Dropout rate"
    )
    parser.add_argument(
        "--filters", "-f", type=int, default=8, help="Number of filters in first conv block"
    )
    args = parser.parse_args()
    print_red(args)
    return args
