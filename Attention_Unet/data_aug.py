import os
import cv2
import numpy as np
import scipy
import scipy.ndimage.filters as filters
import skimage
import skimage.io as io
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

PATH_IMG = "/home/edgar/Documents/Datasets/JB/supervised/imgs/"
PATH_MASK = "/home/edgar/Documents/Datasets/JB/supervised/labels/"


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def rotate_img(path, img_name):
    img = cv2.imread(path + img_name, cv2.IMREAD_UNCHANGED)
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    angles = [90, 180, 270]

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(img, M, (h, w))
        cv2.imwrite(path + str(angle) + "_" + img_name, rotated)


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape
    ).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )

    return scipy.ndimage.interpolation.map_coordinates(
        image, indices, order=1, mode="reflect"
    ).reshape(shape)


def elastic_transform_wrapped(img_path, mask_path):
    im = io.imread(PATH_IMG + img_path)
    im_mask = io.imread(PATH_MASK + mask_path, plugin="tifffile")
    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)

    # im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.09, im_merge.shape[1] * 0.09)
    im_merge_t = elastic_transform(
        im_merge,
        im_merge.shape[1] * 2,
        im_merge.shape[1] * 0.08,
        im_merge.shape[1] * 0.08,
    )  # soft transform

    im_t = im_merge_t[..., 0]
    im_mask_t = im_merge_t[..., 1]
    io.imsave(PATH_IMG + "t_" + img_path, im_t)
    io.imsave(PATH_MASK + "t_" + mask_path, im_mask_t)


if __name__ == "__main__":
    # DA for seg
    # list_img = list_files(PATH_IMG)
    # list_mask = list_files(PATH_MASK)
    #
    # for i, img in enumerate(list_img):
    #     print(img)
    #     mask = list_mask[i]
    #     rotate_img(PATH_IMG, img)
    #     rotate_img(PATH_MASK, mask)

    list_img = list_files(PATH_IMG)
    list_mask = list_files(PATH_MASK)

    for i, img in enumerate(list_img):
        print("transform " + img)
        mask = list_mask[i]
        elastic_transform_wrapped(img, mask)
