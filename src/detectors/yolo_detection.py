import cv2
import numpy as np
import pandas as pd

from src.utils import timeit, draw_boxed_text, read_json


class Detector:
    """Class yolo"""

    @timeit
    def __init__(self):
        self.class_names = read_json('./models/yolo/labels.json')
        self.model = cv2.dnn.readNetFromDarknet(
            # 'models/yolo/yolov3.cfg',
            # 'models/yolo/yolov3.weights')
            'models/yolo/yolov3-tiny.cfg',
            'models/yolo/yolov3-tiny.weights')
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.threshold = 0.3
        self.scale = 0.00392  # 1/255
        self.nms_threshold = 0.4  # Non Maximum Supression threshold
        self.swaprb = True

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1]
                         for i in net.getUnconnectedOutLayers()]
        return output_layers

    @timeit
    def prediction(self, image):
        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), swapRB=self.swaprb, crop=False)
        self.model.setInput(blob)
        output = self.model.forward(self.get_output_layers(self.model))
        return output

    @timeit
    def filter_prediction(self, output, image, conf_th=None, conf_class=None):
        image_height, image_width, _ = image.shape
        df = pd.concat([self.filter_yolo(i) for i in output])
        df = df.assign(
            center_x=lambda x: (x['center_x'] * image_width),
            center_y=lambda x: (x['center_y'] * image_height),
            w=lambda x: (x['w'] * image_width),
            h=lambda x: (x['h'] * image_height),
            x1=lambda x: (x.center_x - (x.w / 2)).astype(int).clip(0),
            y1=lambda x: (x.center_y - (x.h / 2)).astype(int).clip(0),
            x2=lambda x: (x.x1 + x.w).astype(int),
            y2=lambda x: (x.y1 + x.h).astype(int),
            class_name=lambda x: (x['class_id'].astype(int).astype(str).replace(self.class_names)),
            # TODO: python 3.5 fix
            # label=lambda x: (
            #     x.class_name + ': ' + (
            #         x['confidence'].astype(str).str.slice(stop=4)
            #         )
            #     )
        )
        df['label'] = (df['class_name'] + ': ' +
                       df['confidence'].astype(str).str.slice(stop=4))
        cols = ['x1', 'y1', 'w', 'h']
        indices = cv2.dnn.NMSBoxes(
            df[cols].values.tolist(),
            df['confidence'].tolist(), self.threshold, self.nms_threshold)
        if len(indices) > 0:
            df = df.iloc[indices.flatten()]
        return df

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            x_min, y_min, x_max, y_max = box['x1'], box['y1'], box['x2'], box['y2']
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min + 2, 0), max(y_min + 2, 0))
            txt = box['label']
            image = draw_boxed_text(image, txt, txt_loc, color)
        return image

    def filter_yolo(self, chunk):
        pred = np.argmax(chunk[:, 5:], axis=1)
        prob = np.max(chunk[:, 5:], axis=1)
        df = pd.DataFrame(
            np.concatenate(
                [chunk[:, :4], pred.reshape(-1, 1), prob.reshape(-1, 1)],
                axis=1
            ),
            columns=[
                'center_x', 'center_y', 'w', 'h', 'class_id', 'confidence'])
        df = df[df['confidence'] > self.threshold]
        return df


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)

    cv2.imwrite("./imgs/outputcv.jpg", image)
