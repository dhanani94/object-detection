import cv2

from src.utils import draw_boxed_text


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

    def prediction(self, image):
        raise NotImplementedError

    def filter_prediction(self, output, image, conf_th=0.5, conf_class=None):
        raise NotImplementedError
