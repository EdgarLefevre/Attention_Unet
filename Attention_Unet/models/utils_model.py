import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers


def block_down(inputs, filters, drop, w_decay=0.0001, kernel_size=3, name=""):
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        activation="elu",
        name=name + "_conv1",
    )(inputs)
    c = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        activation="elu",
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        name=name + "_conv2",
    )(x)
    p = layers.MaxPooling2D((2, 2), name=name + "_maxpool")(c)
    p = layers.Dropout(drop, name=name + "_dropout")(p)
    return c, p


def bridge(inputs, filters, drop, kernel_size=3):
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="elu",
        name="bridge_conv1",
    )(inputs)
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="elu",
        name="bridge_conv2",
    )(x)
    x = layers.Dropout(drop, name="bridge_dropout")(x)
    return x


def block_up(inputs, conc, filters, drop, w_decay=0.0001, kernel_size=3, name="", attention=False):
    x = layers.Conv2DTranspose(
        filters,
        (2, 2),
        strides=(2, 2),
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        name=name + "_convTranspose",
    )(inputs)
    for i in range(len(conc)):
        if attention:
            gat = gating_signal(inputs, filters)
            att = attention_block(conc[i], gat, filters, name=name+"_att")
            x = layers.concatenate([x, conc[i], att], name=name + "_concatenate" + str(i))
        else:
            x = layers.concatenate([x, conc[i]], name=name + "_concatenate" + str(i))
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        activation="elu",
        name=name + "_conv1",
    )(x)
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        activation="elu",
        name=name + "_conv2",
    )(x)
    x = layers.Dropout(drop, name=name + "_dropout")(x)
    return x


def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input: down-dim feature map
    :param out_size: output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding="same")(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def attention_block(x, gating, inter_shape, name):
    """
    From https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831 ;
    did some adaptation
    """
    theta_x = layers.Conv2D(inter_shape, kernel_size=1, strides=2, padding="same")(
        x
    )  # 8,8,32
    phi_g = layers.Conv2D(inter_shape, kernel_size=1, padding="same")(
        gating
    )  # 16,16,32
    concat_xg = layers.add([phi_g, theta_x])  # 8,8,32
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding="same")(act_xg)  # 8,8,1
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    upsample_psi = layers.UpSampling2D(size=(2, 2), name=name)(sigmoid_xg)  # 16,16,1
    y = layers.multiply([upsample_psi, x])  # 16,16,32
    return y
