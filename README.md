[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fcristianpb%2Fobject-detection%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/cristianpb/object-detection/goto?ref=master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Object detection app (with Taufiq's Modifications)

The frontend Angular application ([strongly based on this repo](https://github.com/cristianpb/object-detection-frontend)) 
is under the [/visual](/visual) directory. The goal of this fork is to focus less on the object direction and more on
the storing/tracking/recording of detections and positions. 

It can use:
* SSD Mobilenet
* Yolo
* Motion detection using OpenCV
* Cascade classifier

## Install
TODO: 

### Raspberry

For Raspberry Pi install OpenCV 4 fast and optimized for this device:

```
git clone https://github.com/dlime/Faster_OpenCV_4_Raspberry_Pi.git
cd Faster_OpenCV_4_Raspberry_Pi/debs
sudo dpkg -i OpenCV*.deb
sudo ldconfig
```

For the dependencies, I prefer to use `.deb` files in Raspberry Pi instead of
`pip` because it doesn't have to compile sources. For installing pandas takes
more than 1 hour using `pip`. More details are in Makefile.

## Utilisation

Once the application is running go to `localhost:5000`

## Used detection models
* [SSD mobilenet](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#use-existing-config-file-for-your-model)
* [Yolo V3](https://pjreddie.com/darknet/yolo/)
