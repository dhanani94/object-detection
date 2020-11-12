#!/usr/bin/env python3
import os
import glob
import cv2
import json
import argparse
from functools import reduce
from importlib import import_module
from itertools import islice
from datetime import datetime
from src.cameras.camera_pi import Camera, Predictor
import logging

from flask import Flask, Response, send_from_directory, request, Blueprint, abort
from src.utils import *

logger = logging.getLogger(__name__)
WIDTH = 320
HEIGHT = 240
BASEURL = '/'
IMAGE_FOLDER = 'imgs'

app = Flask(__name__)

# static html
blueprint_html = Blueprint('html', __name__, url_prefix=BASEURL)


@blueprint_html.route('/', defaults={'filename': 'index.html'})
@blueprint_html.route('/<path:filename>')
def show_pages(filename):
    return send_from_directory('./visual/dist', filename)


app.register_blueprint(blueprint_html)

# API
blueprint_api = Blueprint('api', __name__, url_prefix=BASEURL)


@blueprint_api.route(os.path.join('/', IMAGE_FOLDER, '<path:filename>'))
def image_preview(filename):
    w = request.args.get('w', None)
    h = request.args.get('h', None)
    date = request.args.get('date', None)

    try:
        im = cv2.imread(os.path.join(IMAGE_FOLDER, filename))
        if w and h:
            w, h = int(w), int(h)
            im = cv2.resize(im, (w, h))
        elif date:
            date = (datetime
                    .strptime(date, "%Y%m%d_%H%M%S")
                    .strftime("%d %b %-H:%M")
                    )
            img_h, img_w = im.shape[:-1]
            cv2.putText(
                im, "{}".format(date), (0, int(img_h * 0.98)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return Response(cv2.imencode('.jpg', im)[1].tobytes(),
                        mimetype='image/jpeg')

    except Exception as e:
        logger.error(e)

    return send_from_directory('.', filename)


@blueprint_api.route('/api/delete', methods=['POST'])
def delete_image():
    filename = request.form.get('filename', None)
    try:
        os.remove(filename)
        return json.dumps({'status': filename})
    except Exception as e:
        logger.error(e)
        return abort(404)


def get_data(item):
    if 'pi' in item:
        year = item.split('/')[2][:4]
        month = item.split('/')[2][4:6]
        day = item.split('/')[2][6:8]
        hour = item.split('/')[3][:2]
        minutes = item.split('/')[3][2:4]
        return dict(
            path=item, year=year, month=month, day=day,
            hour=hour, minutes=minutes
        )
    else:
        return dict(path=item)


@blueprint_api.route('/api/images')
def api_images():
    page = int(request.args.get('page', 0))
    page_size = int(request.args.get('page_size', 16))
    mydate = request.args.get('date', None)
    myyear = request.args.get('year', "????")
    mymonth = request.args.get('month', "??")
    myday = request.args.get('day', "??")
    myhour = request.args.get('hour', "??")
    myminutes = request.args.get('minutes', "??")
    mydetection = request.args.get('detected_object', "*")
    if mydate is not None:
        mydate = (datetime
                  .strptime(mydate, "%d/%m/%Y")
                  .strftime("%Y%m%d")
                  )
        myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', mydate, '*.jpg'),
                            recursive=True)
    elif (myyear != "????" or
          mymonth != "??" or
          myday != "??" or
          myhour != "??" or
          myminutes != "??" or
          mydetection != "*"):
        mypath = os.path.join(
            IMAGE_FOLDER, '**',
            f'{myyear}{mymonth}{myday}',
            f'{myhour.zfill(2)}{myminutes}??*{mydetection}*.jpg')
        myiter = glob.iglob(mypath, recursive=True)
    else:
        myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                            recursive=True)

    start = page * page_size
    end = (page + 1) * page_size
    result = [get_data(i) for i in islice(myiter, start, end)]
    logger.debug('->> Start', start, 'end', end, 'len', len(result))
    return json.dumps(result)


@blueprint_api.route('/api/single_image')
def single_image():
    detection = bool(request.args.get('detection', False))
    tracking = bool(request.args.get('tracking', False))
    frame = camera.get_frame()
    if detection:
        annotated_img, df = predictor.prediction(frame, conf_th=0.3, conf_class=[])
        detections, labels = predictor.get_detections(df)
        if "person" in labels:
            f_base_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%f")
            logger.info(f"saving image and label: {f_base_name}")
            cv2.imwrite(f"{detection_dir_img}/{f_base_name}.jpg", frame)
            write_json(f"{detection_dir_label}/{f_base_name}.json", detections)

    elif tracking:
        annotated_img, df = predictor.object_track(frame, conf_th=0.5, conf_class=[1])
    else:
        annotated_img = frame
    return json.dumps(dict(img=img_to_base64(annotated_img), width=WIDTH, height=HEIGHT))


myconditions = dict(
    month=reduce_month,
    year=reduce_year,
    hour=reduce_hour,
    detected_object=reduce_object,
    tracking_object=reduce_tracking,
)


@blueprint_api.route('/api/list_files')
def list_folder():
    condition = request.args.get('condition', 'year')
    myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                        recursive=True)
    newdict = reduce(lambda a, b: myconditions[condition](a, b), myiter, dict())
    # year = item.split('/')[2][:4]
    # month = item.split('/')[2][4:6]
    # day = item.split('/')[2][6:8]
    # hour = item.split('/')[3][:2]
    # minutes = item.split('/')[3][2:4]
    # return json.dumps({k: v for k, v in sorted(newdict.items(), key=lambda item: item[1], reverse=True)})
    return json.dumps(newdict)


# @blueprint_api.route('/api/task/status/<task_id>')
# def taskstatus(task_id):
#     # task = ObjectTracking.AsyncResult(task_id)
#     task = predictor.continous_object_tracking.AsyncResult(task_id)
#     if task.state == 'PENDING':
#         response = {
#             'state': task.state,
#             'object_id': 0,
#         }
#     elif task.state != 'FAILURE':
#         response = {
#             'state': task.state,
#             'object_id': task.info.get('object_id', 0),
#         }
#     else:
#         response = {
#             'state': task.state,
#             'object_id': task.info.get('object_id', 0),
#         }
#     return json.dumps(response)


# @blueprint_api.route('/api/task/launch')
# def launch_object_tracking():
#     task = predictor.object_tracking.delay()
#     # task = predictor.continous_object_tracking.delay()
#     return json.dumps({"task_id": task.id})


# @blueprint_api.route('/api/task/kill/<task_id>')
# def killtask(task_id):
#     response = celery.control.revoke(task_id, terminate=True, wait=True, timeout=10)
#     return json.dumps(response)


# @blueprint_api.route('/api/beat/launch')
# def launch_beat():
#     task = predictor.periodic_capture_continous.delay()
#     return json.dumps({"task_id": task.id})


app.register_blueprint(blueprint_api)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RPi Object Detector')
    parser.add_argument('-p', '--port', help='the port we\'re running on', default=5000, dest='port')
    parser.add_argument('-d', '--detector', help='the target detector [yolo, ssd]', default='yolo', dest='detector')
    parser.add_argument('-i', '--img_dir', help='the image dir [yolo, ssd]', default='./imgs', dest='image_dir')
    parser.add_argument('-v', '--verbose', help='logging level', default=False, dest='verbose', type=bool, const=True,
                        nargs='?')
    parser.add_argument('--cam_rotation', help='camera rotation', default=0, dest='cam_rotation')
    args = parser.parse_args()
    if args.verbose:
        log_level = "debug"
    else:
        log_level = "info"
    initialise_logger(log_level=log_level)

    camera = Camera(args.image_dir, args.cam_rotation)
    detector = import_module(f'src.detectors.{args.detector}_detection').Detector()
    predictor = Predictor(detector, args.image_dir, camera_rotation=args.cam_rotation)

    detection_dir_img = create_dir_if_not_exist(f"{args.image_dir}/detections/images")
    detection_dir_label = create_dir_if_not_exist(f"{args.image_dir}/detections/labels")

    app.run(
        host='0.0.0.0',
        debug=bool(os.getenv('DEBUG')),
        threaded=False,
        port=args.port
    )
