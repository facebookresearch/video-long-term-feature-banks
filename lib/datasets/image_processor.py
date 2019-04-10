# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Image/video clip transformation or augmentation helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import config as cfg
import cv2
import math
import numpy as np


cv2.ocl.setUseOpenCL(False)


def CHW2HWC(image):
    return image.transpose([1, 2, 0])


def HWC2CHW(image):
    return image.transpose([2, 0, 1])


# Image should be in CHW format.
def color_normalization(img, mean, stddev):
    assert len(mean) == img.shape[0], \
        'channel mean not computed properly'
    assert len(stddev) == img.shape[0], \
        'channel stddev not computed properly'
    for idx in range(img.shape[0]):
        img[idx] = img[idx] - mean[idx]
        img[idx] = img[idx] / stddev[idx]
    return img


def pad_image(pad_size, image, order='CHW'):
    if order == 'CHW':
        img = np.pad(
            image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode=str('constant')
        )
    elif order == 'HWC':
        img = np.pad(
            image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode=str('constant')
        )
    return img


def flip_boxes(boxes, im_width):
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
    return boxes


def horizontal_flip_list(prob, images, order='CHW', boxes=None,
                         force_flip=False):

    _, width, _ = images[0].shape
    if np.random.uniform() < prob or force_flip:
        if boxes is not None:
            boxes = flip_boxes(boxes, width)
        if order == 'CHW':
            out_images = []
            for image in images:
                image = np.asarray(image).swapaxes(2, 0)
                image = image[::-1]
                out_images.append(image.swapaxes(0, 2))
            return out_images, boxes
        elif order == 'HWC':
            return [cv2.flip(image, 1) for image in images], boxes
    return images, boxes


def crop_boxes(boxes, x_offset, y_offset):
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset
    return boxes


def random_crop_list(images, size, pad_size=0, order='CHW', boxes=None):
    if pad_size > 0:
        raise NotImplementedError()
        images = [pad_image(pad_size=pad_size, image=image, order=order)
                  for image in images]

    if order == 'CHW':
        if images[0].shape[1] == size and images[0].shape[2] == size:
            return images, boxes
        height = images[0].shape[1]
        width = images[0].shape[2]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [image[:, y_offset:y_offset + size, x_offset:x_offset + size]
                   for image in images]
        assert cropped[0].shape[1] == size, "Image not cropped properly"
        assert cropped[0].shape[2] == size, "Image not cropped properly"
    elif order == 'HWC':
        if images[0].shape[0] == size and images[0].shape[1] == size:
            return images, boxes
        height = images[0].shape[0]
        width = images[0].shape[1]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [image[y_offset:y_offset + size, x_offset:x_offset + size, :]
                   for image in images]
        assert cropped[0].shape[0] == size, "Image not cropped properly"
        assert cropped[0].shape[1] == size, "Image not cropped properly"

    if boxes is not None:
        boxes = crop_boxes(boxes, x_offset, y_offset)
    return cropped, boxes


def center_crop(size, image):
    height = image.shape[0]
    width = image.shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    cropped = image[y_offset:y_offset + size, x_offset:x_offset + size, :]
    assert cropped.shape[0] == size, "Image height not cropped properly"
    assert cropped.shape[1] == size, "Image width not cropped properly"
    return cropped


def spatial_shift_crop_list(size, images, spatial_shift_pos, boxes=None):
    assert spatial_shift_pos in [0, 1, 2]

    height = images[0].shape[0]
    width = images[0].shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_shift_pos == 0:
            y_offset = 0
        elif spatial_shift_pos == 2:
            y_offset = height - size
    else:
        if spatial_shift_pos == 0:
            x_offset = 0
        elif spatial_shift_pos == 2:
            x_offset = width - size

    cropped = [image[y_offset:y_offset + size, x_offset:x_offset + size, :]
               for image in images]
    assert cropped[0].shape[0] == size, "Image height not cropped properly"
    assert cropped[0].shape[1] == size, "Image width not cropped properly"

    if boxes is not None:
        boxes[:, [0, 2]] -= x_offset
        boxes[:, [1, 3]] -= y_offset

    return cropped, boxes


def scale(size, image):
    height = image.shape[0]
    width = image.shape[1]
    if ((width <= height and width == size)
            or (height <= width and height == size)):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image, (new_width, new_height),
        interpolation=getattr(cv2, cfg.INTERPOLATION))
    return img.astype(np.float32)


# Scale the smaller edge of image to size.
def scale_boxes(size, boxes, height, width):
    if ((width <= height and width == size)
            or (height <= width and height == size)):
        return boxes

    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        boxes *= (float(new_height) / height)
    else:
        new_width = int(math.floor((float(width) / height) * size))
        boxes *= (float(new_width) / width)
    return boxes


# Scale the smaller edge of image to a scale from, e.g. 1 / [1/320, 1/256].
# Image should be in HWC format.
def random_short_side_scale_jitter_list(images, min_size,
                                        max_size, boxes=None):

    size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))

    height = images[0].shape[0]
    width = images[0].shape[1]
    if ((width <= height and width == size)
            or (height <= width and height == size)):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width
    return [cv2.resize(image, (new_width, new_height),
                       interpolation=getattr(cv2, cfg.INTERPOLATION)
                       ).astype(np.float32)
            for image in images], boxes


# Image should be in CHW format and BGR channels.
def lighting_list(imgs, alphastd, eigval, eigvec, alpha=None):
    if alphastd == 0:
        return imgs
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(
            eig_val, 3, axis=0),
        axis=1
    )
    out_images = []
    for img in imgs:
        for idx in range(img.shape[0]):
            img[idx] = img[idx] + rgb[2 - idx]
        out_images.append(img)
    return out_images


def blend(image1, image2, alpha):
    return image1 * alpha + image2 * (1 - alpha)


# Image should be in CHW format and BGR channels.
def grayscale(image):
    img_gray = np.copy(image)
    gray_channel = 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]
    img_gray[0] = gray_channel
    img_gray[1] = gray_channel
    img_gray[2] = gray_channel
    return img_gray


def saturation_list(var, images):
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def brightness_list(var, images):
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_bright = np.zeros(image.shape)
        out_images.append(blend(image, img_bright, alpha))
    return out_images


def contrast_list(var, images):
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        img_gray.fill(np.mean(img_gray[0]))
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def color_jitter_list(images, img_brightness=0, img_contrast=0, img_saturation=0):

    jitter = []
    if img_brightness != 0:
        jitter.append('brightness')
    if img_contrast != 0:
        jitter.append('contrast')
    if img_saturation != 0:
        jitter.append('saturation')

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == 'brightness':
                images = brightness_list(img_brightness, images)
            elif jitter[order[idx]] == 'contrast':
                images = contrast_list(img_contrast, images)
            elif jitter[order[idx]] == 'saturation':
                images = saturation_list(img_saturation, images)
    return images
