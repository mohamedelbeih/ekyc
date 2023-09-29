import os
# os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-10.1/lib64"
# /usr/local/cuda/lib64
import argparse
import time
import json
import cv2
import numpy as np
import base64
import glob
import torch
import torchvision
from torch import jit
import os
from math import sqrt
from math import *
import detectron2
from detectron2.utils.logger import setup_logger
import imgaug.augmenters as iaa
import imgaug.augmentables as iaas
import imgaug.augmenters.imgcorruptlike as iaa_corrupt

setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import sys, os, argparse
from detectron2.engine import DefaultTrainer
import math
import io

def get_img_from_base64(base64_string):
    img = base64.b64decode(base64_string.encode('utf-8'))
    #reading image from bytes
    img = np.frombuffer(img, dtype=np.uint8)
    #converting to np data structure
    img = cv2.imdecode(img, flags=1)
    #img = cv2.imread(io.BytesIO(base64.b64decode(base64_string)))
    cv2.imshow("img_show", img)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()
    return img

def get_base64_from_img_path(img_path):
    with open(img_path, "rb") as image_file:
        img_bytes = image_file.read()
        encoded_string = base64.b64encode(img_bytes)
    return encoded_string.decode('ascii')

def resize_image(im, max_side_len=1024):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''

    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(math.ceil(resize_h * ratio))
    resize_w = int(math.ceil(resize_w * ratio))

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    print(resize_h, resize_w)
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im

def flip_ud(image):
    try:
        flipped_image = cv2.flip(image, 0)
    except:
        image = image.astype(np.uint8)  # convert to an unsigned byte
        image *= 255
        flipped_image = cv2.flip(image, 0)
    return flipped_image

def flip_lr(image):
    try:
        flipped_image = cv2.flip(image, 1)
    except:
        image = image.astype(np.uint8)  # convert to an unsigned byte
        image *= 255
        flipped_image = cv2.flip(image, 1)

    return flipped_image

class BaseTransform(object):
    """
    Base class for all transforms, for type-check only.

    Users should never interact with this class.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)

class Transform(BaseTransform):
    """
    A deterministic image transformation, used to implement
    the (probably random) augmentors.

    This class is also the place to provide a default implementation to any
    :meth:`apply_xxx` method.
    The current default is to raise NotImplementedError in any such methods.

    All subclasses should implement `apply_image`.
    The image should be of type uint8 in range [0, 255], or
    floating point images in range [0, 1] or [0, 255]

    Some subclasses may implement `apply_coords`, when applicable.
    It should take and return a nd-array of Nx2, where each row is the (x, y) coordinate.

    The implementation of each method may choose to modify its input data
    in-place for efficient transformation.
    """

    def __init__(self):
        # provide an empty __init__, so that __repr__ will work nicely
        pass

    def __getattr__(self, name):
        if name.startswith("apply_"):
            def f(x):
                raise NotImplementedError("{} does not implement method {}".format(self.__class__.__name__, name))

            return f
        raise AttributeError("Transform object has no attribute {}".format(name))

    def __repr__(self):
        try:
            return _default_repr(self)
        except AssertionError as e:
            log_once(e.args[0], 'warn')
            return super(Transform, self).__repr__()

    __str__ = __repr__

class ResizeTransform(Transform):
    """
    Resize the image.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int):
            new_h, new_w (int):
            interp (int): cv2 interpolation method
        """
        super(ResizeTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        assert img.shape[:2] == (self.h, self.w)
        ret = cv2.resize(
            img, (self.new_w, self.new_h),
            interpolation=self.interp)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


def resize_by_shortest_edge(img, short_edge_length, max_size, interp=cv2.INTER_LINEAR):
    """
    Resize a test image by shortest edge

    Parameters
    ==========

    img : nd-array
        image under test

    short_edge_length : int

    max_size : int
        max side length

    interp : cv2.INTER_LINEAR
        interpolation method

    Returns
    =======

    img : nd-array
          image after resizing
    """
    orig_shape = img.shape[:2]
    # print(orig_shape)

    h, w = orig_shape

    scale = short_edge_length * 1.0 / min(h, w)

    if h < w:
        newh, neww = short_edge_length, scale * w
    else:
        newh, neww = scale * h, short_edge_length

    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale

    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    resizer = ResizeTransform(h, w, newh, neww, interp)
    img = resizer.apply_image(img)

    return img


def clip_boxes(boxes, shape):
    """
    clip boxes to be inside image
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)

def _scale_box(box, scale):
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = x_c - w_half
    scaled_box[2] = x_c + w_half
    scaled_box[1] = y_c - h_half
    scaled_box[3] = y_c + h_half
    return scaled_box

def _paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    assert mask.shape[0] == mask.shape[1], mask.shape

    # This method (inspired by Detectron) is less accurate but fast.

    x0, y0 = list(map(int, box[:2] + 0.5))
    x1, y1 = list(map(int, box[2:] - 0.5))  # inclusive
    x1 = max(x0, x1)  # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret

def get_contr(mask):
    (cnts, hierarchy) = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    approx_cnts = []
    for c in cnts:
        approx = cv2.approxPolyDP(c, 200, True)
        if len(approx) == 4:
            approx_cnts.append(approx)
        else:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            approx_cnts.append(box)
    return approx_cnts

def order_points(pts):
    """
    This function orders the four vertices in clockwise order starting with the tl corner
    """
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def initialze_scripted_model(PATH):
    """
    Initialize torchscript model

    Parameters
    ==========

    PATH : str
           Path of the torchscript model

    Returns
    =======

    img : model object
          loaded torchscripted model
    """

    net = jit.load(PATH)
    return net


def initialize_detectron_model(PATH, device="cpu", output_dir='output'):
    """
    This function initializes a model and first set the cfg parameters used in training

    Parameters
    ==========

    PATH : str
        path of the saved model

    device : str
        cpu or gpu

    Returns
    =======
    model : pytorch model

    """
    print("/".join(PATH.split("/")[:-1]) + "/cfg.yaml")
    cfg = get_cfg()

    cfg.merge_from_file("/".join(PATH.split("/")[:-1]) + "/cfg.yaml")

    ##############################################
    cfg.MODEL.WEIGHTS = PATH  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = '/media/res13/db3ef46c-0099-481c-a38f-0542989db3551/ID_prespective_correction/output_medium_num/model_final.pth' #os.path.join( "output", "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=200
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST=13 #*3
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold



    predictor = DefaultPredictor(cfg)

    return predictor, cfg.dump()


def inference_model(img, model):
    """
    Load, preprocess, inference, post-processing for the given image

    Parameters
    ==========

    img : nd-array
          image input as array

    model: model object
           loaded torchscripted model

    Returns
    =======

    boxes: nd-array
           array contains the bounding boxes of the predicted instances

    labels : nd-array
             array contains the class labels for the predicted instances

    masks : nd-array
            array contains the predicted masks instances of shape (X,28,28)

    scores : nd-array
             array contains the score confidences for each predicted instance
    """
    t1=time.time()
    outputs = model(
        img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    timex=time.time()-t1
    masks = outputs["instances"].to("cpu").pred_masks.numpy()
    scores = outputs["instances"].to("cpu").scores.numpy()
    labels = outputs["instances"].to("cpu").pred_classes.numpy()
    boxes = outputs["instances"].to("cpu").pred_boxes

    # print(labels)
    return boxes, labels, masks, scores,timex


def side_classify(labels, scores):
    """
    check the classified side of the given ID depending on the highest confidence

    Parameters
    ==========

    labels : nd-array
             Array of labels indexes predicted by model

    scores : nd-array
             Array of score confidence for each instance

    Returns
    =======

    id_: int
          0 label for Front ID
          1 label for Back ID
          2 label for Non ID

    index_id_mask: int
                   index of the mask of the card itself
    """
    pred_class_map = {22: 2, 0: 0, 11: 1}

    labels = list(labels)
    # get type of id
    # check if id is face

    if 0 in labels:
        # check if side is front
        if 22 in labels:
            id_ = pred_class_map[0] if scores[labels.index(0)] > scores[labels.index(22)] else pred_class_map[22]
            index_id_mask = labels.index(0) if scores[labels.index(0)] > scores[labels.index(22)] else labels.index(22)
        else:
            id_ = pred_class_map[0]
            index_id_mask = labels.index(0)

    elif 11 in labels:
        # check if side is back
        if 22 in labels:
            id_ = pred_class_map[11] if scores[labels.index(11)] > scores[labels.index(22)] else pred_class_map[22]
            index_id_mask = labels.index(11) if scores[labels.index(11)] > scores[labels.index(22)] else labels.index(
                22)
        else:
            id_ = pred_class_map[11]
            index_id_mask = labels.index(11)
    else:

        id_ = 2
        index_id_mask = 0

    return id_, index_id_mask


def correct_rotation(card_poly, lines_poly, lines_fields, side):
    """
    Find the if ID is rotated and the angle to rotate with to correct the ID position, also find if the ID is flipped based on the rotation
    and the position of the right lines in the normal ID all that using the lines detection and classification from the detectron output.

    Parameters
    ==========

    card_poly: nd-array
                contains the 4-points of the card in original image

    lines_poly: nd-array
                contains the 4-points of each field of the card related to the original image

    lines_fields: list
                  contains int label for each field

    side: int
          value that determine the side of the card 0 for front and 1 for back

    Returns
    =======

    rotation_type: int
                 a number represent the rotation needed to correct the ID position.
                 0: the images need no rotation
                 1: the image need rotation 90 degree clockwise
                 2: the image need rotation 90 degree counterclockwise
                 3: the image need rotation 180 degree

    flipping_type: int
                 a number represent the position of right ID lines in the image depending on the rotation_type.
                 rotation_type == 0 or 3 and flipping_type == 0: the lines is closer to the right side
                 rotation_type == 0 or 3 and flipping_type == 1: the lines is closer to the left side
                 rotation_type == 1 or 2 and flipping_type == 0: the lines is closer to the upper side
                 rotation_type == 1 or 2 and flipping_type == 1: the lines is closer to the lower side
    """

    rotation_type = 0
    if len(lines_poly) < 2: return rotation_type, 0  # we need at least 2 lines

    upper_side = sqrt((card_poly[0][0] - card_poly[1][0]) ** 2 + (card_poly[0][1] - card_poly[1][1]) ** 2)
    bottom_side = sqrt((card_poly[2][0] - card_poly[3][0]) ** 2 + (card_poly[2][1] - card_poly[3][1]) ** 2)
    left_side = sqrt((card_poly[1][0] - card_poly[2][0]) ** 2 + (card_poly[1][1] - card_poly[2][1]) ** 2)
    right_side = sqrt((card_poly[0][0] - card_poly[3][0]) ** 2 + (card_poly[0][1] - card_poly[3][1]) ** 2)

    horizental_edge = (upper_side + bottom_side) / 2
    virtical_edge = (left_side + right_side) / 2

    upper_line = 0
    while (upper_line not in lines_fields and upper_line < 10): upper_line += 1

    lower_line = 9
    while (lower_line not in lines_fields and lower_line > 0): lower_line -= 1

    if upper_line >= 10 or lower_line <= 0:
        return 0, 0

    upper_line = lines_fields.index(upper_line)
    lower_line = lines_fields.index(lower_line)

    if horizental_edge > virtical_edge:  # angle 0 or 180
        ymin_1, ymax_1 = min(lines_poly[upper_line, :, 1]), max(lines_poly[upper_line, :, 1])
        ymin_2, ymax_2 = min(lines_poly[lower_line, :, 1]), max(lines_poly[lower_line, :, 1])
        y_1 = (ymin_1 + ymax_1) / 2
        y_2 = (ymin_2 + ymax_2) / 2

        if y_1 < y_2:  # angle 0
            rotation_type = 0
        else:  # angle 180
            rotation_type = 3


    else:  # 90 or -90 angle
        xmin_1, xmax_1 = min(lines_poly[upper_line, :, 0]), max(lines_poly[upper_line, :, 0])
        xmin_2, xmax_2 = min(lines_poly[lower_line, :, 0]), max(lines_poly[lower_line, :, 0])
        x_1 = (xmin_1 + xmax_1) / 2
        x_2 = (xmin_2 + xmax_2) / 2

        if x_1 < x_2:  # left
            rotation_type = 1
        else:  # right
            rotation_type = 2

    # flipping type
    if side == 0: test_lines = [0, 1, 2, 3, 4, 5, 6]
    if side == 1: test_lines = [0, 2, 3, 4, 7, 8, 9]
    # print("side",side)
    test_line = 0
    while (test_lines[test_line] not in lines_fields and test_line < len(test_lines)): test_line += 1
    # if test_line== len(test_lines):
    #     break
    test_line = lines_fields.index(test_lines[test_line])
    line = lines_poly[test_line]
    flipping_type = 0

    if rotation_type == 0 or rotation_type == 3:
        # test with x axis
        card_width = max(card_poly[:, 0]) - min(card_poly[:, 0])
        x_in_card = line[:, 0] - min(card_poly[:, 0])
        x_left = min(x_in_card)
        x_right = min(card_width - x_in_card)
        if x_right < x_left:
            flipping_type = 0  # lines closer to the right side
        else:
            flipping_type = 1  # lines closer to left side
    else:
        # test with x axis
        card_hight = max(card_poly[:, 1]) - min(card_poly[:, 1])
        y_in_card = line[:, 1] - min(card_poly[:, 1])
        y_upper = min(y_in_card)
        y_lower = min(card_hight - y_in_card)
        if y_upper < y_lower:
            flipping_type = 0  # lines closer to the upper side
        else:
            flipping_type = 1  # lines closer to lower side

    return rotation_type, flipping_type


def rotate_ID_card(img, card_poly, rotation_type, flipping_type):
    """
    Function that takes information about rotation_type and flipping_type from correct_rotation function
    and correct the image rotation and flipping

    Parameters
    ==========

    img: nd-array
         the original input image

    card_poly: nd-array
                contains the 4-points of the card in original image

    rotation_type: int
                 a number represent the rotation needed to correct the ID position.
                 0: the images need no rotation
                 1: the image need rotation 90 degree clockwise
                 2: the image need rotation 90 degree counterclockwise
                 3: the image need rotation 180 degree

    flipping_type: int
                 a number represent the position of right ID lines in the image depending on the rotation_type.
                 rotation_type == 0 or 3 and flipping_type == 0: the lines is closer to the right side
                 rotation_type == 0 or 3 and flipping_type == 1: the lines is closer to the left side
                 rotation_type == 1 or 2 and flipping_type == 0: the lines is closer to the upper side
                 rotation_type == 1 or 2 and flipping_type == 1: the lines is closer to the lower side

    Returns
    =======

    img: nd-array
         the original input image after correcting the rotation and flipping in the image

    card_poly: nd-array
                contains the 4-points of the card in original image after correcting the rotation and flipping in the image

    """

    if rotation_type == 0:
        if flipping_type == 1:
            # flip_lr
            img = flip_lr(img)
            card_poly[:, 0] = img.shape[1] - card_poly[:, 0]

    elif rotation_type == 1:  # rotate 90 clockwise
        if flipping_type == 1:
            # flip_up
            img = flip_ud(img)
            card_poly[:, 1] = img.shape[0] - card_poly[:, 1]

        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        tmp = card_poly[:, 0].copy()
        card_poly[:, 0] = img.shape[1] - card_poly[:, 1]
        card_poly[:, 1] = tmp

    elif rotation_type == 2:  # rotate 90 counter clockwise
        if flipping_type == 0:
            # flip_up
            img = flip_ud(img)
            card_poly[:, 1] = img.shape[0] - card_poly[:, 1]

        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        tmp = card_poly[:, 1].copy()
        card_poly[:, 1] = img.shape[0] - card_poly[:, 0]
        card_poly[:, 0] = tmp

    elif rotation_type == 3:  # rotate 180
        if flipping_type == 0:
            # flip_lr
            img = flip_lr(img)
            card_poly[:, 0] = img.shape[1] - card_poly[:, 0]

        img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        card_poly[:, 1] = img.shape[0] - card_poly[:, 1]
        card_poly[:, 0] = img.shape[1] - card_poly[:, 0]

    card_poly = order_points(card_poly)

    return img, card_poly


def rotate_ID(img, card_poly, lines_poly, masks, rotation_type, flipping_type):
    """
    Function that takes information about rotation_type and flipping_type from correct_rotation function
    and correct the image rotation and flipping and also the polys that output from the detectron model

    Parameters
    ==========

    img: nd-array
         the original input image

    card_poly: nd-array
                contains the 4-points of the card in original image

    lines_poly: nd-array
                contains the 4-points of each field of the card related to the original image

    masks : list
            list of masks corresponds to the given labels

    rotation_type: int
                 a number represent the rotation needed to correct the ID position.
                 0: the images need no rotation
                 1: the image need rotation 90 degree clockwise
                 2: the image need rotation 90 degree counterclockwise
                 3: the image need rotation 180 degree

    flipping_type: int
                 a number represent the position of right ID lines in the image depending on the rotation_type.
                 rotation_type == 0 or 3 and flipping_type == 0: the lines is closer to the right side
                 rotation_type == 0 or 3 and flipping_type == 1: the lines is closer to the left side
                 rotation_type == 1 or 2 and flipping_type == 0: the lines is closer to the upper side
                 rotation_type == 1 or 2 and flipping_type == 1: the lines is closer to the lower side

    Returns
    =======

    img: nd-array
         the original input image after correcting the rotation and flipping in the image

    card_poly: nd-array
                contains the 4-points of the card in original image after correcting the rotation and flipping in the image

    lines_poly: nd-array
                contains the 4-points of each field of the card related to the original image after correcting the rotation and flipping in the image

    masks : list
            contains a list of masks after flipping and axis rotation
    """

    if rotation_type == 0:
        if flipping_type == 1:
            # flip_lr
            img = flip_lr(image=img)
            card_poly[:, 0] = img.shape[1] - card_poly[:, 0]
            lines_poly[:, :, 0] = img.shape[1] - lines_poly[:, :, 0]
            for idx, each_mask in enumerate(masks):
                masks[idx] = flip_lr(image=each_mask)

    elif rotation_type == 1:  # rotate 90 clockwise
        if flipping_type == 1:
            # flip_up
            img = flip_ud(image=img)
            card_poly[:, 1] = img.shape[0] - card_poly[:, 1]
            lines_poly[:, :, 1] = img.shape[0] - lines_poly[:, :, 1]
            for idx, each_mask in enumerate(masks):
                masks[idx] = flip_ud(image=each_mask)

        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        for idx, each_mask in enumerate(masks):
            try:
                each_mask = each_mask.astype(np.uint8)  # convert to an unsigned byte
                each_mask *= 255
                masks[idx] = cv2.rotate(each_mask, cv2.cv2.ROTATE_90_CLOCKWISE)
            except:
                print("mask 1 error")
                # cv2.imshow("mask",each_mask)
                # cv2.waitKey()
        tmp = card_poly[:, 0].copy()
        card_poly[:, 0] = img.shape[1] - card_poly[:, 1]
        card_poly[:, 1] = tmp

        tmp = lines_poly[:, :, 0].copy()
        lines_poly[:, :, 0] = img.shape[1] - lines_poly[:, :, 1]
        lines_poly[:, :, 1] = tmp


    elif rotation_type == 2:  # rotate 90 counter clockwise
        if flipping_type == 0:
            # flip_up
            img = flip_ud(image=img)
            card_poly[:, 1] = img.shape[0] - card_poly[:, 1]
            lines_poly[:, :, 1] = img.shape[0] - lines_poly[:, :, 1]
            for idx, each_mask in enumerate(masks):
                masks[idx] = flip_ud(image=each_mask)

        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        for idx, each_mask in enumerate(masks):
            try:
                each_mask = each_mask.astype(np.uint8)  # convert to an unsigned byte
                each_mask *= 255
                masks[idx] = cv2.rotate(each_mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            except:
                print("masks 2 error")
                # cv2.imshow("mask",each_mask)

        tmp = card_poly[:, 1].copy()
        card_poly[:, 1] = img.shape[0] - card_poly[:, 0]
        card_poly[:, 0] = tmp

        tmp = lines_poly[:, :, 1].copy()
        lines_poly[:, :, 1] = img.shape[0] - lines_poly[:, :, 0]
        lines_poly[:, :, 0] = tmp


    elif rotation_type == 3:  # rotate 180
        if flipping_type == 0:
            # flip_lr
            img = flip_lr(image=img)
            card_poly[:, 0] = img.shape[1] - card_poly[:, 0]
            lines_poly[:, :, 0] = img.shape[1] - lines_poly[:, :, 0]
            for idx, each_mask in enumerate(masks):
                masks[idx] = flip_lr(image=each_mask)
        img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        for idx, each_mask in enumerate(masks):
            masks[idx] = cv2.rotate(each_mask, cv2.cv2.ROTATE_180)
        card_poly[:, 1] = img.shape[0] - card_poly[:, 1]
        card_poly[:, 0] = img.shape[1] - card_poly[:, 0]

        lines_poly[:, :, 1] = img.shape[0] - lines_poly[:, :, 1]
        lines_poly[:, :, 0] = img.shape[1] - lines_poly[:, :, 0]

    card_poly = order_points(card_poly)
    for i, line in enumerate(lines_poly):
        lines_poly[i] = order_points(line)

    return img, card_poly, lines_poly, masks


def get_filled_line(image, mask, line_poly):
    """
    crop the field from the image using the mask pixels and fill the rest of the rectangle with inpainted color of the mask edge

    Parameters
    ==========

    image : nd-array
            original image after processing and consider as a source of cropping

    mask : nd-array
           same size of the image with the same processing but contains the mask of the field only

    line_poly : nd-array
                4-points of the field in the original image that the edges of the mask will be inpainted

    Returns
    =======

    field_image : nd-array
                  the field image with inpainted mask edge
    """

    y1 = int(max(line_poly[0][1], 0))
    y2 = int(line_poly[2][1])
    x1 = int(max(line_poly[0][0], 0))
    x2 = int(line_poly[2][0])
    line_image = image[y1:y2, x1:x2]
    line_mask = mask[y1:y2, x1:x2]
    line_mask_3d = line_mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    line_image = line_image * line_mask_3d / 255
    inpaint_mask = 255 - line_mask
    image = image.astype(np.uint8)
    field_image = cv2.inpaint(line_image.astype(np.uint8), inpaint_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return field_image


def get_boundaries(masks, labels, card_id_mask, side):
    """
    get the boundaries of the instances relative to the original image

    Parameters
    ==========

    masks : nd-array
             Array of full masks of each instance

    labels : nd-array
             Array of labels indexes predicted by model

    card_id_mask: int
                  index of the id card mask

    side : int
           integer determines the side of the given card 0 for front and 1 for back

    Returns
    =======

    card_poly: nd-array
               contains the 4-points of the card in original image

    lines_poly: nd-array
                 contains the 4-points of each field of the card related to the original image

    lines_fields: list
                  contains int label for each field
                  if front side = 0
                    2: name-1
                    3: name-2
                    4: address-1
                    5: address-2
                    6: ID-Num
                    9: FCN
                  if back side = 1
                    0: ID-Num
                    1: release-date
                    2: job-1
                    3: job-2
                    4: gender
                    5: religion
                    6: marital_status
                    7: husband_name
                    8: Producer_num
                    9: end-date

    lines_masks : list
                  field masks corresponding to the lines_fields and lines_poly

    rotation_type : int
                    integer in range [0,3]
                    0: the images need no rotation
                    1: the image need rotation 90 degree clockwise
                    2: the image need rotation 90 degree counterclockwise
                    3: the image need rotation 180 degree

    flipping_type: int
                   a number represent the position of right ID lines in the image depending on the rotation_type.
                   rotation_type == 0 or 3 and flipping_type == 0: the lines is closer to the right side
                   rotation_type == 0 or 3 and flipping_type == 1: the lines is closer to the left side
                   rotation_type == 1 or 2 and flipping_type == 0: the lines is closer to the upper side
                   rotation_type == 1 or 2 and flipping_type == 1: the lines is closer to the lower side
    """

    pred_id_mask = masks[card_id_mask]
    card_cntr = get_contr(pred_id_mask)

    lines_cntrs = []
    for label, mask in zip(labels, masks):

        if label in [0, 22, 11]:  # check fields label only
            continue
        cntr = get_contr(mask)
        lines_cntrs.append([label, cntr, mask])

    card_poly = card_cntr
    card_poly = card_poly[0].reshape((4, 2))
    card_poly = order_points(card_poly)

    lines_poly = []
    field_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
                 12: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 7, 20: 8, 21: 9}
    lines_fields = []
    lines_masks = []
    for item in lines_cntrs:
        label, poly, mask = item
        lines_fields.append(field_map[label])
        line = poly[0].reshape(4, 2)
        line = order_points(line)
        lines_poly.append(line)
        lines_masks.append(mask)

    lines_poly = np.array(lines_poly)
    rotation_type, flipping_type = correct_rotation(card_poly, lines_poly, lines_fields, side)

    return card_poly, lines_poly, lines_fields, lines_masks, rotation_type, flipping_type

def get_card_boundaries(masks, labels, card_id_mask, side):
    """
    get the boundaries of the instances relative to the original image for the cards only regardless the fields positions


    Parameters
    ==========

    masks : nd-array
             Array of full masks of each instance

    labels : nd-array
             Array of labels indexes predicted by model

    card_id_mask: int
                  index of the id card mask

    side : int
           integer determines the side of the given card 0 for front and 1 for back

    Returns
    =======

    card_poly: nd-array
               contains the 4-points of the card in original image


    """

    pred_id_mask = masks
    card_cntr = get_contr(pred_id_mask)

    card_poly = card_cntr
    card_poly = card_poly[0].reshape((4, 2))
    card_poly = order_points(card_poly)

    return card_poly

def check_fields(lines_fields, card_side, masks, field_points, width=500, height=300):
    """
    Function to save lines detected in text files

    Parameters
    ==========

    lines_fields: list
                  contains int label for each field

    card_side: int
               value that determine the side of the card 0 for front and 1 for back

    masks: list
           contains the masks for each field after rotation, flipping and skewing

    field_points: list
                  contain the 4-points of each field related to the original image after rotation, flipping and skewing

    width: int (optional)
           max x of the fields which is the width of the image

    height: int (optional)
           max y of the fields which is the heights of the image

    Returns
    =======

    total_fields: list
                  contains all expected needed fields for IDs in the form of  [field,4points,flag,mask] flag to determine whether to crop from the original or the warped
    """

    temp_face = [[136, 80, 500, 110],
                 [135, 110, 500, 140],
                 [136, 140, 500, 175],
                 [148, 170, 500, 205],
                 [190, 230, 500, 275],
                 [44, 268, 176, 294]]

    # template of desired back fields of the id, arranged from top to down
    # these are their locations in a 500*300 tempalte image

    temp_back = [[229, 15, 410, 45],
                 [132, 15, 225, 45],
                 [147, 45, 410, 70],
                 [146, 70, 410, 95],
                 [360, 90, 410, 125],
                 [277, 90, 332, 125],
                 [165, 90, 229, 125],
                 [149, 110, 404, 145],
                 [430, 140, 488, 170],
                 [149, 135, 420, 175]]

    total_fields = []
    if card_side == 0:
        interest_fields = [2, 3, 4, 5, 6, 9]
        template_fields = temp_face
    else:
        interest_fields = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        template_fields = temp_back

    for i_field, field in enumerate(interest_fields):
        if field in lines_fields:
            poly_field = lines_fields.index(field)
            coor = field_points[poly_field]
            orig_h, orig_w = masks[poly_field].shape
            y1 = max(coor[0][1], 0)
            y2 = min(coor[2][1], orig_h)
            x1 = max(coor[0][0], 0)
            x2 = min(coor[2][0], orig_w)

            if ((y2 - y1 <= 5) or (x2 - x1 <= 5)) and field != 8:
                coor = template_fields[i_field]
                new_coor = np.array([coor[0], coor[1], coor[2], coor[1], coor[2], coor[3], coor[0], coor[3]]).reshape(
                    (4, 2))
                total_fields.append([field, new_coor, 0, 0])
                continue

            total_fields.append([field, coor, 1, masks[poly_field] * 255])

        else:

            if (field == 3 or field == 7 or field == 6) and (card_side == 1):
                continue
            coor = template_fields[i_field]
            new_coor = np.array([coor[0], coor[1], coor[2], coor[1], coor[2], coor[3], coor[0], coor[3]]).reshape(
                (4, 2))
            total_fields.append([field, new_coor, 0, 0])

    return total_fields


def transform_to_warped_img(orig_img, card_poly, lines_poly, trans_height=300, trans_width=500):
    """
    transform the points of the card polygon to the given image height and width, also transform the fields points to the generated warped image

    Parameters
    ==========

    orig_img : nd-array
               original image to be transformed

    card_poly: nd-array
               contains the 4-points of the card in original image

    lines_poly: nd-array
                 contains the 4-points of each field of the card related to the original image

    trans_height: int (optional)
                  height of the resultant warped image

    trans_width: int (optional)
                 width of the resultant warped image
    Returns
    =======

    warped_img : nd-array
                 generated resultant warped image of the given card

    trans_lines_poly: nd-array
                      contains the 4-points of each field of the card related to the warped image

    """
    dst = np.array([
        [0, 0],
        [trans_width - 1, 0],
        [trans_width - 1, trans_height - 1],
        [0, trans_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(card_poly,
                                    dst)  # computing the matrix that can transform CORNER pointS from the source to distination

    warped_img = cv2.warpPerspective(orig_img, M, (trans_width, trans_height))

    trans_lines_poly = []
    if len(lines_poly != 0):
        trans_lines_poly = cv2.perspectiveTransform(lines_poly, M)
        trans_lines_poly = trans_lines_poly.reshape(-1, 4, 2).astype('int')

    return warped_img, trans_lines_poly


def save_results(path, warped_id, img, side, total_lines_poly, out_path):
    """
    Function to save warped image and lines detected as images

    Parameters
    ==========

    path : str
        path of input image

    warped_id : nd-array
                warped ID image

    img : nd-array
          original image after rotation, flipping and de-skewing

    side: int
          label represent the side of ID 0: Front and 1:Back

    total_lines_poly: list
                      list of fields polygons with its class as [field_label,points]
    out_path : str
               output dir paths

    """

    filename = ".".join(path.split("/")[-1].split(".")[:-1])
    os.system("mkdir -p " + out_path + "/" + filename)
    fields_map_front = {0: "Gomhoryt_Misr", 1: "Betaka_Title", 2: "Name-1", 3: "Name-2", 4: "Address-1", 5: "Address-2",
                        6: "ID-NUM", 7: "Image", 8: "birth-date", 9: "FCN"}
    fields_map_back = {0: "ID-NUM", 1: "release-date", 2: "Job-1", 3: "Job-2", 4: "gender", 5: "religion",
                       6: "marital-status", 7: "husband-name", 8: "prod-num", 9: "end-date"}
    if side == 0:
        write_map = fields_map_front
    else:
        write_map = fields_map_back
    if warped_id is not None:
        cv2.imwrite(out_path + "/" + filename + "/warped_image.bmp", warped_id)
    cv2.imwrite(out_path + "/" + filename + "/input_image.bmp", img)
    if True:
        for field, points, detected, field_mask in total_lines_poly:
            if detected:
                line_image = get_filled_line(img, field_mask, points)
            else:
                y1 = int(points[0][1])
                y2 = int(points[2][1])
                x1 = int(points[0][0])
                x2 = int(points[2][0])

                line_image = warped_id[y1:y2, x1:x2]
            cv2.imwrite(out_path + "/" + filename + "/" + write_map[field] + ".bmp", line_image)
    # except:
    #     print("total line poly ",total_lines_poly)
    return


def transform_image(img, card_points, trans_height=300, trans_width=500):
    """
    warp the given image with the card points to the given height and width

    Parameters
    ==========

    img : nd-array
          image input as array

    card_points: nd-array
                 array of 4 points and must take the shape of (4,2)

    trans_height: int (optional)
                  height of the resultant warped image

    trans_width: int (optional)
                 width of the resultant warped image

    Returns
    =======

    warped_img : nd-array
                 generated resultant warped image of the given card

    flag: bool
          determine if the process success or not True if failed

    """

    if card_points.shape != (4, 2):
        return None, True

    card_points = order_points(card_points)
    h, w, _ = img.shape
    min_x = card_points[0][0]
    min_y = card_points[0][1]
    max_x = card_points[2][0]
    max_y = card_points[2][1]
    if min_y < 0 or min_x < 0 or max_x > w or max_y > h:
        return None, True

    dst = np.array([
        [0, 0],
        [trans_width - 1, 0],
        [trans_width - 1, trans_height - 1],
        [0, trans_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(np.asarray(card_points, dtype="float32"), dst)

    warped_img = cv2.warpPerspective(img, M, (trans_width, trans_height))

    return warped_img, False


def getAngleBetweenPoints(bottom_left, bottom_right):
    """
    Calculate the angle between two points with refernce to x-axis

    Parameters
    ==========

    bottom_left : list
                  list of 2 points corresponds to the bottom left point

    bottom_right : list
                   list of 2 points corresponds to the bottom right point

    Returns
    =======

    angle : float
            floating number representing the degrees of the card base
    """
    deltaY = float(bottom_right[1]) - float(bottom_left[1])
    deltaX = float(bottom_right[0]) - float(bottom_left[0])
    trunc_ang = atan2(deltaY, deltaX)
    while trunc_ang < 0.0:
        trunc_ang += pi * 2

    angle = degrees(trunc_ang)
    return angle


def rotate_image(image, points, angle):
    """
    Rotate the given image and points with the given angle returning the rotated points and rotated image

    Parameters
    ==========

    image : nd-array
            image input as array

    points : nd-array
             array of 4 points and must take the shape of (4,2)

    angle : float
            angle of card skewing in degrees

    Returns
    =======

    image : nd-array
            rotated image with the given angle

    points : nd-array
             rotated points with the given angle that determines the card in the rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    points = cv2.transform(np.array([points]), rot_mat)[0]
    return result, points


def get_expand_image(img, card_points):
    """
    calculate the angle of the card in the given image from the given points, then crop the de-skewed card after expanding the co-ordinates by 5% of the image size to be considered as input image to the second inferencing
    Parameters
    ==========

    img : nd-array
          image input as array

    card_points: nd-array
                 array of 4 points and must take the shape of (4,2)

    Returns
    =======

    expanded_img : nd-array
                   expanded image after de-skewing containing the card in the middle
    """
    angle = getAngleBetweenPoints(card_points[3], card_points[2])
    if angle == 0.0:
        img_rotated = img
    else:
        img_rotated, card_points = rotate_image(img, card_points, angle)

    h, w, _ = img_rotated.shape
    h_off = int(h * 0.05)
    w_off = int(w * 0.05)

    min_x = int(min(max(card_points[0][0] - w_off, 0), max(card_points[3][0] - w_off, 0)))
    max_x = int(max(min(card_points[1][0] + w_off, w), min(card_points[2][0] + w_off, w)))
    min_h = int(min(max(card_points[0][1] - h_off, 0), max(card_points[1][1] - h_off, 0)))
    max_h = int(max(min(card_points[2][1] + h_off, h), min(card_points[3][1] + h_off, h)))

    expanded_img = img_rotated[min_h:max_h, min_x:max_x, :]

    return expanded_img


def get_angle_rotate_image_points(img, points, mask_list, points_list):
    """
    Rotate the given images and corresponding points with the given angle returning the rotated images and the new points relative to it

    Parameters
    ==========

    img : nd-array
          image input as array

    points : nd-array
             corresponding point to img - array of 4 points and must take the shape of (4,2) for the card points

    mask_list : list
                list of masks images for each detected field

    points_list : list
                  list of points for each detected field


    Returns
    =======

    rot_img : nd-array
              rotated image with the given angle

    points : nd-array
             rotated points with the given angle that determines the card in the rotated image

    image_list : list
                 rotated masks for each field

    list_points : list
                  rotated points relative to the rotated masks
    """
    angle = getAngleBetweenPoints(points[3], points[2])
    if angle == 0.0:
        return img, points, mask_list, points_list
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    card_points = cv2.transform(np.array([points]), rot_mat)[0]

    for idx, points in enumerate(points_list):
        new_mask = cv2.warpAffine(mask_list[idx].astype(np.uint8), rot_mat, mask_list[idx].shape[1::-1],
                                  flags=cv2.INTER_LINEAR)
        points = cv2.transform(np.array([points]), rot_mat)[0]
        points_list[idx] = points
        mask_list[idx] = new_mask
    return rot_img, card_points, mask_list, points_list


def check_mandatory_fields(fields_class, side):
    """
    Check that all mandatory fields are detected by the model or not

    Parameters
    ==========

    fields_class: list
                  contains int label for each field
                  if front side = 0
                    2: name-1
                    3: name-2
                    4: address-1
                    5: address-2
                    6: ID-Num
                    9: FCN
                  if back side = 1
                    0: ID-Num
                    1: release-date
                    2: job-1
                    3: job-2
                    4: gender
                    5: religion
                    6: marital_status
                    7: husband_name
                    8: Producer_num
                    9: end-date

    side : int
           determine the side of the card 0 for front and 1 for back

    Returns
    =======

    flag : bool
           if True means that all mandatory fields detected, and if false means there is at least one mandatory field not detected

    """
    front_mandatory = {2, 3, 4, 5, 6, 9}
    back_mandatory = {0, 1, 2, 4, 5, 8, 9}
    if side == 0:
        mandatory_fields = front_mandatory
    else:
        mandatory_fields = back_mandatory

    diff = mandatory_fields - set(fields_class)
    if len(diff) == 0:
        return True
    return False

def prespective_transform(corner_pts,img,output_dir):
    """
    this function apply perspective transformation to detected area then save it
    """
    #perspective transform
    # Locate points of the documents
	# or object which you want to transform
    corner_pts = np.asarray(corner_pts,dtype="int64")
    pts1 = np.float32(corner_pts)
    pts2 = np.float32([[0, 0], [550, 0],[550, 85],[0, 85]])
            
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    trans_img = cv2.warpPerspective(img, matrix, (550, 85))  
            
    return trans_img

def segment_detected_area(trans_img,image_id,saving_dir):

    img_gray = cv2.cvtColor(trans_img, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV) ######
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,2)) ####
    mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)  

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    contours = contours[::-1]
    
    pts_of_correct_areas = []
    for cntr in contours:
        
        corner_pts = order_points(cntr.reshape(cntr.shape[0],cntr.shape[2]))
        corner_pts = np.asarray(corner_pts,dtype="int64")
        
        upper_side = sqrt((corner_pts[0][0] - corner_pts[1][0]) ** 2 + (corner_pts[0][1] - corner_pts[1][1]) ** 2)
        bottom_side = sqrt((corner_pts[2][0] - corner_pts[3][0]) ** 2 + (corner_pts[2][1] - corner_pts[3][1]) ** 2)
        left_side = sqrt((corner_pts[1][0] - corner_pts[2][0]) ** 2 + (corner_pts[1][1] - corner_pts[2][1]) ** 2)
        right_side = sqrt((corner_pts[0][0] - corner_pts[3][0]) ** 2 + (corner_pts[0][1] - corner_pts[3][1]) ** 2)
        
        horizental_edge = (upper_side + bottom_side) / 2
        virtical_edge = (left_side + right_side) / 2

        if (horizental_edge / virtical_edge > 550/50 ) and (horizental_edge / virtical_edge < 550/6) :
            pts_of_correct_areas.append(corner_pts)

    for i in range(len(pts_of_correct_areas)):
        
        pts1 = np.float32(pts_of_correct_areas[i])

        # edit y of each pt to give more space while cropping
        pts1[0][1] = pts1[0][1]-4
        pts1[1][1] = pts1[1][1]-4
        pts1[2][1] = pts1[2][1]+4
        pts1[3][1] = pts1[3][1]+4

        pts2 = np.float32([[0, 0], [1375, 0],[1375, 60],[0, 60]])
        # Apply Perspective Transform Algorithm
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(trans_img, matrix, (1375, 60))
        cv2.imwrite(saving_dir +"/" + str(i) + "_" +image_id,result)

def run_pipeline(path, model, output_dir):
    """
    run pipeline considers as a caller function for the default pipeline of the ID preprocessor without any manual interactions

    Parameters
    ==========

    path : str
           path of input image

    model : pytorch object
            loaded model that will be used in inferencing

    output_dir: str
                path to the output directroy that will be created to dump results of the pipeline

    """
    img = cv2.imread(path)

    boxes, labels, masks, scores = inference_model(img, model)

    side_id, card_idx_mask = side_classify(labels, scores)

    if side_id == 2:
        return

    card_points, fields_points, fields_class, fields_masks, rotation_type, flipping_type = get_boundaries(masks, labels,
                                                                                                          card_idx_mask,
                                                                                                          side_id)
    img, card_points = rotate_ID_card(img, card_points, rotation_type, flipping_type)
    img = get_expand_image(img, card_points)

    boxes, labels, masks, scores = inference_model(img, model)
    side_id, card_idx_mask = side_classify(labels, scores)

    if side_id == 2:
        return

    card_points, fields_points, fields_class, fields_masks, rotation_type, flipping_type = get_boundaries(masks, labels,
                                                                                                          card_idx_mask,
                                                                                                          side_id)
    img, card_points, fields_points, fields_masks = rotate_ID(img, card_points, fields_points, fields_masks,
                                                              rotation_type, flipping_type)

    img_rotated, card_points, rotated_masks, rotated_fields = get_angle_rotate_image_points(img, card_points,
                                                                                            fields_masks, fields_points)

    mand_check = check_mandatory_fields(fields_class, side_id)

    if mand_check:
        warped_card = None
    else:
        warped_card, flag = transform_image(img_rotated, card_points)

    total_fields = check_fields(fields_class, side_id, rotated_masks, rotated_fields)

    save_results(path, warped_card, img_rotated, side_id, total_fields, output_dir)

    return

def draw_rectangle_using_4lines(x1,y1,x2,y3,c,image):
    # printx1,x2,x3,x4,y1,y2,y3,y4)
    # printc)
    if c==0:
        color=0
    else:
        color=255
    if True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print"-=============-")
        # printy1,y3)

        cv2.rectangle(image,(x1,y1),(x2,y3),[92,255,0],7)
        cv2.putText(image, f"{c}", (x1, y1+50), font, 3, (0,0, 255), 3, cv2.LINE_AA)

def run_pipeline_custom(path, model, output_dir, card_points=None, side=None):
    """
    run customized pipeline considers as a caller function in case of manual interaction for a given 4 points of the card or/and
    the side of the given image

    Parameters
    ==========

    path : str
           path of input image

    model : pytorch object
            loaded model that will be used in inferencing

    output_dir: str
                path to the output directroy that will be created to dump results of the pipeline

    card_points: nd-array (array)
                 array of 4 points and must take the shape of (4,2)

    side: str (optional)
          must be one of the following (Front, Back) else it will be neglected

    """
    img = cv2.imread(path)
    print(path)
    #img = resize_image(img, 300)

    boxes, labels, masks, scores ,timex= inference_model(img, model)

    
    #print(time.time()-t1)
 

    measure_speed=False
    results_dir = '/'.join(path.split('/')[:-1])+'_results'
    img_id = ".".join(path.split('/')[-1].split('.')[:-1])
    os.makedirs(results_dir,exist_ok=True)
    os.system("cp "+path +" " +results_dir)
    if scores.shape[0] >= 1:
        best_class_index = np.where(scores == np.amax(scores,axis=0))[0][0]
        with open(results_dir+'/'+img_id+'.txt','w') as f:
            #for i, box in enumerate(boxes):
                
                        
            corner_pts = get_card_boundaries(masks[best_class_index], labels[best_class_index], 0, None)
            pts = ''
            for x in corner_pts.flatten():
                pts = pts + str(int(x)) + ','
            
            f.write(pts + str(labels[best_class_index])+'\n')  
    else:
        with open(results_dir+'/'+img_id+'.txt','w') as f:
            f.write("\n") 

        #cv2.imwrite(output_dir + "/" + path.split("/")[-1],img)


    # save_results(path, warped_card, img, side_id, total_fields, output_dir)
    return 1,timex


from exceptions import Validation
def base64_to_bytes(base64_img,error_code):
    """
    This function converts base64 images to an bytes file.

    Parameters
    ===========
        base64_img: str
            The base64 string representing the image.
        error_code: int
            The error code of the raised ValidationError if the string couldn't be converted.

    Returns
    ========
        bytes
            The image bytes after converting

    Raises
    =======
        Validation
            if the base64 string cannot be converted, our custom Validation error will be raised.

    """
    try:
        return base64.b64decode(base64_img)

    except (ValueError, TypeError):
        raise Validation(error_code)

def run_pipeline(img, model):
    """
    detect card label

    Parameters
    ==========

    path : str
           image bytes

    model : pytorch object
            loaded model that will be used in inferencing

    """
    #reading image from bytes
    img = np.frombuffer(img, dtype=np.uint8)
    #converting to np data structure
    img = cv2.imdecode(img, flags=1)
    boxes, labels, masks, scores ,timex= inference_model(img, model)    
    
    label2img_dict = {}
    for i, box in enumerate(boxes):
        corner_pts = get_card_boundaries(masks[i], labels[i], 0, None)
        corner_pts = corner_pts.flatten()
        xs = [corner_pts[0],corner_pts[2],corner_pts[4],corner_pts[6]]
        xs = [ int(x) for x in xs ]
        ys = [corner_pts[1],corner_pts[3],corner_pts[5],corner_pts[7]]
        ys = [ int(y) for y in ys ]
        min_x,max_x,min_y,max_y = min(xs) , max(xs) , min(ys) , max(ys)
        label2img_dict[labels[i]] = img[min_y:max_y,min_x:max_x]    
    return label2img_dict                

if __name__ == "__main__":
    model, _ = initialize_detectron_model("detectron_model/"+sys.argv[1]+"/model_final.pth")
    with open("detectron_model/" + sys.argv[1] + "/label_config.json") as f:
        detectron_id_2_label = json.load(f)
        detectron_id_2_label = {detectron_id_2_label[x]['detectron_id']:x for x in detectron_id_2_label}
    with open("img.json") as f:
        image_dict = json.load(f)
    image = base64_to_bytes(image_dict['img'],"4010")
    labelint2img_dict = run_pipeline(image, model)
    labelname2img_dict = {}
    for l in labelint2img_dict:
        field_image = labelint2img_dict[l]
        #converting from cv2 image to base64
        # retval, buffer = cv2.imencode('.jpg', field_image)
        # jpg_as_text = base64.b64encode(buffer)
        base64_string = base64.b64encode(cv2.imencode('.jpg', field_image)[1]).decode()
        labelname2img_dict[detectron_id_2_label[l]] = base64_string

    with open("labelname2img_dict.json","w") as f:
        f.write(json.dumps(labelname2img_dict,indent=4))
        
    
