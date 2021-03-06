import cv2
import numpy as np
import pandas as pd

from src.detectors.base_detector import BaseDetector
from src.utils import timeit, read_json


class Detector(BaseDetector):
    """Class ssd"""

    @timeit
    def __init__(self):
        super().__init__()
        self.swaprb = True
        self.class_names = read_json('./models/ssd_mobilenet/labels.json')
        self.model = cv2.dnn.readNetFromTensorflow(
            'models/ssd_mobilenet/frozen_inference_graph.pb',
            'models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        self.colors = np.random.uniform(0, 255, size=(100, 3))

    @timeit
    def prediction(self, image):
        self.model.setInput(
            cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=self.swaprb))
        output = self.model.forward()
        result = output[0, 0, :, :]
        return result

    @timeit
    def filter_prediction(self, output, image, conf_th=0.5, conf_class=None):
        if not conf_class:
            conf_class = []
        height, width = image.shape[:-1]
        df = pd.DataFrame(
            output,
            columns=[
                '_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        df = df.assign(
            x1=lambda x: (x['x1'] * width).astype(int).clip(0),
            y1=lambda x: (x['y1'] * height).astype(int).clip(0),
            x2=lambda x: (x['x2'] * width).astype(int),
            y2=lambda x: (x['y2'] * height).astype(int),
            class_name=lambda x: (x['class_id'].astype(int).astype(str).replace(self.class_names)),
            confidence=lambda x: (x['confidence'])
        )
        df['label'] = (df['class_name'] + ': ' +
                       df['confidence'].astype(str).str.slice(stop=4))
        df = df[df['confidence'] > conf_th]
        if len(conf_class) > 0:
            df = df[df['class_id'].isin(conf_class)]
        return df

    @staticmethod
    def get_detection_dict(df):
        output = []
        for idx, box in df.iterrows():
            output.append({
                "points": [(box['x1'], box['y1']), (box['x2'], box['y2'])],
                "label": box['label']
            })
        return output
