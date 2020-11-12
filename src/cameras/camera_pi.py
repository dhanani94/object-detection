import glob
import io
import os
import time
from datetime import datetime
from functools import reduce

import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

from src.cameras.base_camera import BaseCamera
from src.centroidtracker import CentroidTracker
from src.utils import reduce_tracking


class Camera(BaseCamera):

    def frames(self):
        with PiCamera() as camera:
            camera.rotation = int(str(self.camera_rotation))
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg',
                                               use_video_port=True):
                # return current frame
                stream.seek(0)
                _stream = stream.getvalue()
                data = np.fromstring(_stream, dtype=np.uint8)
                img = cv2.imdecode(data, 1)
                yield img

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()


class Predictor:
    """Docstring for Predictor. """

    def __init__(self, detector, image_dir, im_width=640, im_height=480, camera_rotation=0):
        self.im_width = im_width
        self.im_height = im_height
        self.image_dir = image_dir
        self.detector = detector
        self.camera_rotation = camera_rotation
        self.ct = CentroidTracker(maxDisappeared=20)

    def prediction(self, img, conf_th=0.3, conf_class=None):
        if not conf_class:
            conf_class = []
        output = self.detector.prediction(img)
        df = self.detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = self.detector.draw_boxes(img, df)
        return img

    def object_track(self, img, conf_th=0.3, conf_class=None):
        if not conf_class:
            conf_class = []
        output = self.detector.prediction(img)
        df = self.detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = self.detector.draw_boxes(img, df)
        boxes = df[['x1', 'y1', 'x2', 'y2']].values
        objects = self.ct.update(boxes)
        if len(boxes) > 0 and (df['class_name'].str.contains('person').any()):
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        return img

    def object_tracking(self):
        myiter = glob.iglob(os.path.join(self.image_dir, '**', '*.jpg'),
                            recursive=True)
        newdict = reduce(lambda a, b: reduce_tracking(a, b), myiter, dict())
        startID = max(map(int, newdict.keys()), default=0) + 1
        ct = CentroidTracker(startID=startID)
        with PiCamera() as camera:
            camera.resolution = (1280, 960)  # twice height and widht
            camera.rotation = int(str(os.environ['CAMERA_ROTATION']))
            camera.framerate = 10
            with PiRGBArray(camera, size=(self.im_width, self.im_height)) as output:
                while True:
                    camera.capture(output, 'bgr', resize=(self.im_width, self.im_height))
                    img = output.array
                    result = self.detector.prediction(img)
                    df = self.detector.filter_prediction(result, img)
                    img = self.detector.draw_boxes(img, df)
                    boxes = df[['x1', 'y1', 'x2', 'y2']].values
                    previous_object_ID = ct.nextObjectID
                    objects = ct.update(boxes)
                    if len(boxes) > 0 and (
                            df['class_name'].str.contains('person').any()) and previous_object_ID in list(
                        objects.keys()):
                        for (objectID, centroid) in objects.items():
                            text = "ID {}".format(objectID)
                            cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                        day = datetime.now().strftime("%Y%m%d")
                        directory = os.path.join(self.image_dir, 'pi', day)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        ids = "-".join(list([str(i) for i in objects.keys()]))
                        hour = datetime.now().strftime("%H%M%S")
                        filename_output = os.path.join(
                            directory, "{}_person_{}_.jpg".format(hour, ids)
                        )
                        cv2.imwrite(filename_output, img)
                    time.sleep(0.300)

    def capture_continous(self):
        with PiCamera() as camera:
            camera.resolution = (1280, 960)  # twice height and widht
            camera.rotation = self.camera_rotation
            camera.framerate = 10
            with PiRGBArray(camera, size=(self.im_width, self.im_height)) as output:
                camera.capture(output, 'bgr', resize=(self.im_width, self.im_height))
                image = output.array
                result = self.detector.prediction(image)
                df = self.detector.filter_prediction(result, image)
                if len(df) > 0:
                    if (df['class_name']
                            .str
                            .contains('person|bird|cat|wine glass|cup|sandwich')
                            .any()):
                        day = datetime.now().strftime("%Y%m%d")
                        directory = os.path.join(self.image_dir, 'pi', day)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        image = self.detector.draw_boxes(image, df)
                        classes = df['class_name'].unique().tolist()
                        hour = datetime.now().strftime("%H%M%S")
                        filename_output = os.path.join(directory, "{}_{}_.jpg".format(hour, "-".join(classes)))
                        cv2.imwrite(filename_output, image)
