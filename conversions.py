from functools import reduce
import cv2
import numpy as np
from pycocotools import mask as mask_utils

def cvat_rle_to_binary_image_mask(cvat_rle: dict, img_h: int, img_w: int) -> np.ndarray:
    # convert CVAT tight object RLE to COCO-style whole image mask
    rle = cvat_rle['rle']
    left = cvat_rle['left']
    top = cvat_rle['top']
    width = cvat_rle['width']

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    value = 0
    offset = 0
    for rle_count in rle:
        while rle_count > 0:
            y, x = divmod(offset, width)
            mask[y + top][x + left] = value
            rle_count -= 1
            offset += 1
        value = 1 - value

    return mask


def binary_image_mask_to_cvat_rle(image: np.ndarray) -> dict:
    # convert COCO-style whole image mask to CVAT tight object RLE

    istrue = np.argwhere(image == 1).transpose()
    top = int(istrue[0].min())
    left = int(istrue[1].min())
    bottom = int(istrue[0].max())
    right = int(istrue[1].max())
    roi_mask = image[top:bottom + 1, left:right + 1]

    # compute RLE values
    def reduce_fn(acc, v):
        if v == acc['val']:
            acc['res'][-1] += 1
        else:
            acc['val'] = v
            acc['res'].append(1)
        return acc
    roi_rle = reduce(
        reduce_fn,
        roi_mask.flat,
        { 'res': [0], 'val': False }
    )['res']

    cvat_rle = {
        'rle': roi_rle,
        'top': top,
        'left': left,
        'width': right - left + 1,
        'height': bottom - top + 1,
    }

    return cvat_rle

def cvat_rle_to_coco_rle(cvat_rle: dict, img_h: int, img_w: int) -> dict:
    # covert CVAT tight object RLE to COCO whole image mask RLE
    binary_image_mask = cvat_rle_to_binary_image_mask(cvat_rle, img_h=img_h, img_w=img_w)
    return mask_utils.encode(np.asfortranarray(binary_image_mask))


def deserialize_cvat_rle(serialized_cvat_rle: dict) -> dict:
    return {
        'rle': list(map(int, serialized_cvat_rle['rle'].split(','))),
        'top': int(serialized_cvat_rle['top']),
        'left': int(serialized_cvat_rle['left']),
        'width': int(serialized_cvat_rle['width']),
        'height': int(serialized_cvat_rle['height']),
    }

def serialize_cvat_rle(cvat_rle: dict) -> dict:
    return {
        'rle': ', '.join(map(str, cvat_rle['rle'])),
        'top': str(cvat_rle['top']),
        'left': str(cvat_rle['left']),
        'width': str(cvat_rle['width']),
        'height': str(cvat_rle['height']),
    }

def create_circle_mask(image_shape, center, radius):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask