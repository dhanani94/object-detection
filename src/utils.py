"""Utilities for logging."""
import base64
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np

ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

logger = logging.getLogger(__name__)


def initialise_logger(log_level="info", output_dir="logs"):
    console_log_level = logging.INFO

    if log_level == "debug":
        console_log_level = logging.DEBUG

    date_fmt = '%Y-%m-%d %H:%M:%S'
    p_format = '%(asctime)s.%(msecs)03d %(name)-12s: %(levelname)s: %(message)s'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = "{}/log.txt".format(output_dir)

    console_handler = logging.StreamHandler()
    logging.basicConfig(level=console_log_level, format=p_format, datefmt=date_fmt, handlers=[console_handler])
    main_logger = logging.getLogger('')

    # Add file handler to the root main_logger
    file_handler = RotatingFileHandler(output_file, maxBytes=50000000, backupCount=5)
    file_log_fmt = logging.Formatter(fmt=p_format, datefmt=date_fmt)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_log_fmt)
    main_logger.addHandler(file_handler)


def read_json(filename):
    with open(filename, 'r') as json_data:
        return json.load(json_data)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger = logging.getLogger(method.__name__)
        logger.debug('{} {:.3f} sec'.format(method.__name__, te - ts))
        return result

    return timed


def img_to_base64(img):
    """encode as a jpeg image and return it"""
    buffer = cv2.imencode('.jpg', img)[1].tobytes()
    jpg_as_text = base64.b64encode(buffer)
    base64_string = jpg_as_text.decode('utf-8')
    return base64_string


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin + 1, h - margin - 2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1] + h, topleft[0]:topleft[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


def reduce_month(accu, item):
    if 'pi' not in item:
        return accu
    year = item.split('/')[2][:4]
    if year not in accu:
        accu[year] = dict()
    month = item.split('/')[2][4:6]
    if month in accu[year]:
        accu[year][month] += 1
    else:
        accu[year][month] = 1
    return accu


def reduce_year(accu, item):
    if 'pi' not in item:
        return accu
    year = item.split('/')[2][:4]
    if year in accu:
        accu[year] += 1
    else:
        accu[year] = 1
    return accu


def reduce_hour(accu, item):
    if 'pi' not in item:
        return accu
    condition = item.split('/')[3][:2]
    if condition in accu:
        accu[condition] += 1
    else:
        accu[condition] = 1
    return accu


def reduce_object(accu, item):
    if 'pi' not in item:
        return accu
    condition = item.split('/')[3].split('_')[1].split('-')
    for val in condition:
        if val in accu:
            accu[val] += 1
        else:
            accu[val] = 1
    return accu


def reduce_tracking(accu, item):
    if 'pi' not in item:
        return accu
    condition = item.split('/')[3].split('_')[2].split('-')
    for val in condition:
        if val in accu:
            accu[val] += 1
        else:
            accu[val] = 1
    return accu
