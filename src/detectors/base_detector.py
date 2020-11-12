import cv2
import numpy as np
import pandas as pd

from src.utils import timeit, draw_boxed_text, read_json


class BaseDetector:

    def __init__(self):
        self.colors = []
        pass

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            x_min, y_min, x_max, y_max = box['x1'], box['y1'], box['x2'], box['y2']
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min + 2, 0), max(y_min + 2, 0))
            txt = box['label']
            image = draw_boxed_text(image, txt, txt_loc, color)
        return image

    @staticmethod
    def get_detection_dict(df):
        output = []
        for idx, box in df.iterrows():
            output.append({
                "points": [(box['x1'], box['y1']), (box['x2'], box['y2'])],
                "label": box['label']
            })
        return

    def prediction(self, image):
        raise NotImplementedError

    def filter_prediction(self, output, image, conf_th=0.5, conf_class=None):
        raise NotImplementedError
