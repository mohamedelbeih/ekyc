import argparse
import time
import cv2
import numpy as np
import os
import glob
import torch
from torch import jit
import os
from math import sqrt
from math import *
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import sys, os, argparse
from detectron2.engine import DefaultTrainer
import math


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

device = torch.device('cpu')
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
    min_edge_len = 300  # shortest edge to be used in resize
    max_edge_len = 1024  # maximum edge to be used in resize

    orig_shape = img.shape[:2]
    resized_img = resize_by_shortest_edge(img, min_edge_len, max_edge_len)

    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])

    out = model(torch.from_numpy(np.transpose(resized_img, (2, 0, 1))).to(device))

    boxes = out[0].cpu().detach().numpy()

    labels = out[1].cpu().detach().numpy()

    masks = out[2].cpu().detach().numpy().reshape((-1, 28, 28))

    scores = out[3].cpu().detach().numpy()

    # scale the predicted bounding boxes to the original image
    boxes = boxes / scale

    # make sure that the scaled bounding boxes fit on the original image size to make the minimum value is zero and max value is h,w of the original image
    boxes = clip_boxes(boxes, orig_shape)

    # resize masks to original shape
    if masks is not None:
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks)]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    return boxes, labels, masks, scores

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

    x0, y0 = list(map(int, box[:2]))
    x1, y1 = list(map(int, box[2:]))  # inclusive
    x1 = max(x0, x1)  # require at least 1x1
    y1 = max(y0, y1)

    w = x1 - x0
    h = y1 - y0

    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')

    ret[y0:y1, x0:x1] = mask
    return ret
    
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
      
def detect_document(img, model):
    """
    document type detection pipeline

    Parameters
    ==========

    path : str
           path of input image

    model : pytorch object
            loaded model that will be used in inferencing

    output_dir: str
                path to the output directroy that will be created to dump results
                of the pipeline


    """
    boxes, labels, masks, scores = inference_model(img, model)

    if scores.shape[0] >= 1: # means that there is a document in the image
        # getting the most likely document frompredicted documents score
        i = np.argmax(scores)
        corner_pts = get_card_boundaries(masks[i], labels[i], 0, None)
        img = cv2.polylines(img, [corner_pts.astype('int')],True, (0, 255, 0), 2) 
        card = True
        return card , corner_pts
        
    else:
        card = False
        return card , None
import json
with open("Document_Aliveness_verification/card_classifier/class_map.json") as f:
    id2type = json.load(f)
     
def class_doc(img, model):
    """
    document type detection pipeline

    Parameters
    ==========

    path : str
           path of input image

    model : pytorch object
            loaded model that will be used in inferencing

    output_dir: str
                path to the output directroy that will be created to dump results
                of the pipeline


    """
    boxes, labels, masks, scores = inference_model(img, model)

    if scores.shape[0] >= 1: # means that there is a document in the image
        # getting the most likely document frompredicted documents score
        i = np.argmax(scores)
        doc_id = labels[i]
        return id2type[str(doc_id)]
    else:
        return False
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default='dev',
                        help='path of directory of files')
    parser.add_argument('--model', type=str,
                        default='models/v.1/model.ts',
                        help='saved model dir of mask_rcnn')
    opt = parser.parse_args()
    
    model_path = opt.model  #adapted_from_idefy_ts

    print("sting")

    st = time.time()
    model = initialze_scripted_model(model_path)
    os.system("rm -r " + opt.path + "/*_res")
    count = 0
 
    for path in glob.glob(opt.path + "/*"):  # (opt.path +"/*"):
        if True : #path.split(".")[-1] in ["jpg", "JPG", "jpeg", "PNG", "JPEG", "png", "BMP", "bmp"]:
            try:
                print(path)
                img = cv2.imread(path)
                out = detect_document(img,model)
                count += 1
                cv2.imwrite("Output/"+str(count)+".jpg",out)
            except Exception as e:
                print("error in path", path)
                print("error name ", e)
                print("exception done")
                print("-------------")
        end = time.time()
        print("elapsed time = ",end-st)

