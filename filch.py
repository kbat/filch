#!/usr/bin/env python
#
# Raspberry Pi AI Camera (IMX500) Model Zoo:
# https://github.com/raspberrypi/imx500-models
#

import os, sys
from pathlib import Path
import tomllib
import logging
from multiprocessing import Pool
# #from functools import lru_cache
from functools import cache

import argparse
import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

from libcamera import Transform
import requests
import time

from suntime import Sun
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from tzlocal import get_localzone_name

import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

logger = logging.getLogger(__name__)


pool = Pool(processes=1)

class GracefulKiller:
        """ https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully """
        kill_now = False
        def __init__(self):
                signal.signal(signal.SIGINT, self.exit_gracefully)
                signal.signal(signal.SIGTERM, self.exit_gracefully)

        def exit_gracefully(self, signum, frame):
                self.kill_now = True

class Detection:
        def __init__(self, coords, category, conf, metadata, imx500, picam2):
                """Create a Detection object, recording the bounding box, category and confidence."""
                self.category = category
                self.conf = conf
                self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model file",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()

class Filch:
        """ A Surveillance System Based on an RPi AI Camera

        Daylight setup: the camera is operating between sunrise and sunset
        """

        def __init__(self, args):
                print(f"Filch class constructor")
                self.args = args
                latitude = 55.52
                longitude = 13.11
                self.sun = Sun(latitude, longitude)
                zone_name = get_localzone_name()
                self.tz = ZoneInfo(zone_name)
                self.model = self.args.model

                self.imx500 = None
                self.config = None
                self.intrinsics = None

                self.camera_start()

                self.last_results = None
                self.previous_objects = []
                self.current_objects = []
                self.picam2.pre_callback = self.draw_detections
                logger.info("Starting the loop")
                self.labels = self.get_labels()

                self.last_detections = []
                self.objects_to_ignore = ('spoon')
                self.objects_to_follow = ('person', 'cat', 'horse')

                self.database      = None
                self.ntfy_channel  = None
                self.url           = None

        #@lru_cache
        @cache
        def get_labels(self):
                labels = self.intrinsics.labels

                if self.intrinsics.ignore_dash_labels:
                        labels = [label for label in labels if label and label != "-"]
                return labels

        def override_intrinsics(self):
                """ Override intrinsics from args """
                # This must be called before instantiation of Picamera2
                print("\t Overriding intrinsics...")
                self.imx500 = IMX500(self.model)
                self.intrinsics = self.imx500.network_intrinsics
                if not self.intrinsics:
                        self.intrinsics = NetworkIntrinsics()
                        self.intrinsics.task = "object detection"
                elif self.intrinsics.task != "object detection":
                        print("Network is not an object detection task", file=sys.stderr)
                        exit()

                for key, value in vars(self.args).items():
                        if key == 'labels' and value is not None:
                                with open(value, 'r') as f:
                                        self.intrinsics.labels = f.read().splitlines()
                        elif hasattr(self.intrinsics, key) and value is not None:
                                setattr(self.intrinsics, key, value)

                # Defaults
                if self.intrinsics.labels is None:
                        with open("/home/kbat/projects/test1/assets/coco_labels.txt", "r") as f:
                                self.intrinsics.labels = f.read().splitlines()
                self.intrinsics.update_with_defaults()

                if self.args.print_intrinsics:
                        print(self.intrinsics)
                        exit()

        def camera_start(self):
                """ Initialisation of the RPI AI camera """
                print("Initialisation of the RPI AI camera")
                self.override_intrinsics()

                self.picam2 = Picamera2(self.imx500.camera_num)
                #     config = self.picam2.create_preview_configuration(
                #             controls={"FrameRate": self.intrinsics.inference_rate},
                #             buffer_count=3, #12
                #             queue=False,
                #             transform=Transform(hflip=True, vflip=True),
                # #            main={"size": (4056, 3040)},
                # #            main={"size": (1920, 1439)},
                # #            main={"size": (1200, 900)},
                #     )

                timelapse_config = self.picam2.create_still_configuration(
                        main={"size": (2028, 1520)},
                        buffer_count=2,
                        queue=False, # you always get the most recent frame
                        transform=Transform(hflip=True, vflip=True),
                        # FrameRate is irrelevant for single stills, but fine to
                        # set (btw, self.intrinsics.inference_rate is 26)
                        controls={"FrameRate": 5},
                )

                self.config = timelapse_config

                print("Configuration:",self.config)
                self.picam2.align_configuration(self.config)
                print("Aligned configuration:",self.config)

                #    self.imx500.show_network_fw_progress_bar()
                self.picam2.start(self.config, show_preview=False)

                if self.intrinsics.preserve_aspect_ratio:
                        self.imx500.set_auto_aspect_ratio()

                # Give the camera a moment to adjust exposure
                # suggestion (might heko): increase from 0.5 to 5 sec if the first image is too dark
                time.sleep(0.5)

        def camera_stop(self):
                """ Stop camera  """
                self.picam2.stop()
                self.picam2.close()

                del self.picam2
                self.picam2 = 0

                del self.imx500
                self.imx500 = 0


        def loop(self):
                """ Surveillance loop """

                time_prev=0 # previous time counter (for timelapse)
                timelapse_period = 10*60
                sleep_time = 2 # seconds

                killer = GracefulKiller()

                while not killer.kill_now:
                    N = self.get_time_to_sunrise()
                    if N:
                            self.camera_stop()
                            time.sleep(N)
                            # self.picam2.start(config, show_preview=False)
                            self.camera_start()

                    self.last_results = self.parse_detections(self.picam2.capture_metadata())
                    self.current_objects = tuple(self.labels[int(r.category)] for r in self.last_results)
            #        self.current_objects = tuple(map(lambda o: o.val, t))
                    if self.current_objects:
                        logger.info(f"{self.current_objects=}")
                    msg = []
                    for obj in self.current_objects:
                         if obj not in self.previous_objects:
            #                 logger.debug(obj)
                             msg.append(str(obj))

                    nmsg = len(msg)
                    follow = self.isFollow(self.current_objects)

                    if nmsg or follow:
                       path = self.create_today_folder()
                       jpg=self.get_timestamp()+"-"+"-".join(msg).replace(" ", "_")+".jpg"
                       jpgpath = os.path.join(path, jpg)
                       self.picam2.capture_file(jpgpath)
                       if nmsg and not self.isIgnore(self.current_objects):
                               msg = ", ".join(msg)
                               self.ntfy(msg, jpgpath.replace(self.database, ""))

                    self.previous_objects = self.current_objects

                    if not follow and sleep_time:
                            logger.debug(f"Sleeping for {sleep_time} sec.")
                            time.sleep(sleep_time)

                    time_now = time.time()
                    if int(time_now - time_prev) > timelapse_period:
                            path = self.create_today_folder()
                            jpg=self.get_timelapse_timestamp()+"-timelapse.jpg"
                            jpgpath = os.path.join(path, jpg)
                            logger.debug(f"Capture timelapse into {jpgpath}")

                            self.picam2.capture_file(jpgpath)

                            time_prev = time_now

                logger.info("Stopping the loop (killed)")
                self.camera_stop()

        def get_time_to_sunrise(self):
                """Return 0 if it's daytime now.

                Otherwise return numer of seconds to the soonest sunrise.

                """
                sun = self.sun
                tz = self.tz

                today = date.today()
                tomorrow = today + timedelta(days=1)

                sunset_today = sun.get_local_sunset_time(today)
                sunrise_today = sun.get_local_sunrise_time(today)
                sunrise_tomorrow = sun.get_local_sunrise_time(tomorrow)

                now = datetime.now(tz)

                if now < sunrise_today:
                        next_sunrise = sunrise_today
                else:
                        next_sunrise = sunrise_tomorrow

                delay = timedelta(minutes=30)

                if now > sunrise_today and now < sunset_today + delay:
                        return 0
                else:
                        time_to_sunrise = int((next_sunrise-now).total_seconds())

                        logger.info(f"Dark time -> sleeping until sunrise for {time_to_sunrise} sec.")
                        return time_to_sunrise

        def draw_detections(self, request, stream="main"):
            """Draw the detections for this request onto the ISP output."""
            detections = self.last_results
            if detections is None:
                return
            labels = self.get_labels()

            with MappedArray(request, stream) as m:
                for detection in detections:
                    x, y, w, h = detection.box
                    label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

                    # Calculate text size and position
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_x = x + 5
                    text_y = y + 15

                    # Create a copy of the array to draw the background with opacity
                    overlay = m.array.copy()

                    # Draw the background rectangle on the overlay
                    cv2.rectangle(overlay,
                                  (text_x, text_y - text_height),
                                  (text_x + text_width, text_y + baseline),
                                  (255, 255, 255),  # Background color (white)
                                  cv2.FILLED)

                    alpha = 0.30
                    cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

                    # Draw text on top of the background
                    cv2.putText(m.array, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Draw detection box
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=1)

                if self.intrinsics.preserve_aspect_ratio:
                    b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
                    color = (255, 0, 0)  # red
                    cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


        def get_timestamp(self):
                """ Return current time with milliseconds """
                now = datetime.now()
                time_str = now.strftime("%y%m%d-%H%M%S")
                time_with_ms = f"{time_str}-{now.microsecond // 1000:03d}"
                return time_with_ms

        def get_timelapse_timestamp(self):
                """ Return current time without milliseconds """
                now = datetime.now()
                time_str = now.strftime("%y%m%d-%H%M%S")
                return time_str

        def get_date(self):
                """ Return current date as a YYMMDD string."""
                return datetime.now().strftime("%y%m%d")

        def create_today_folder(self):
                """ Create a folder for today's images if it does not exist and return its path. """
                today_folder = self.get_date()
                path = f"{self.database}/{today_folder}"
                folder = Path(path)
                folder.mkdir(parents=True, exist_ok=True)
                return path

        def parse_detections(self, metadata: dict):
            """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
            bbox_normalization = self.intrinsics.bbox_normalization
            bbox_order = self.intrinsics.bbox_order
            threshold = self.args.threshold
            iou = self.args.iou

            np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
            input_w, input_h = self.imx500.get_input_size()

            if np_outputs is None:
                    self.last_detections = []
                    return self.last_detections

            if self.intrinsics.postprocess == "nanodet":
                boxes, scores, classes = \
                    postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                                  max_out_dets=self.args.max_detections)[0]
                from picamera2.devices.imx500.postprocess import scale_boxes
                boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            else:
                boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
                if bbox_normalization:
                    boxes = boxes / input_h

                if bbox_order == "xy":
                    boxes = boxes[:, [1, 0, 3, 2]]
                boxes = np.array_split(boxes, 4, axis=1)
                boxes = zip(*boxes)

            self.last_detections = [
                Detection(box, category, score, metadata, self.imx500, self.picam2)
                for box, score, category in zip(boxes, scores, classes)
                if score > threshold
            ]
            return self.last_detections

        def send(self, msg, url):
                now = datetime.now()
                print("sending", now)
                requests.post(f"https://ntfy.sh/{self.ntfy_channel}", data=f"{msg}".encode(encoding='utf-8'),
                           headers={"Actions": f"view, Open, {url}"})
                print(" sent", datetime.now()-now)


        def ntfy(self, msg, jpg):
            url=f"{self.url}{jpg}"
        #    now = datetime.now()
        #    pool.apply_async(send, (msg, url))
        #    print("async sent", datetime.now() - now)
            if self.ntfy_channel:
                    self.send(msg, url)
                    print(msg + f" {url}")
            else:
                    print("WARNING: No ntfy.sh channel is defined -> not sending")

        def isIgnore(self, objects):
                ignore = set(self.objects_to_ignore) & set(objects)
                if ignore:
                        logger.info(f"Ignoring {ignore}")
                return ignore

        def isFollow(self, objects):
                follow = set(self.objects_to_follow) & set(objects)
                if follow:
                        logger.info(f"Following {follow}")
                elif objects:
                        logger.info(f"Not following {objects}")
                return follow

def main():
        config_file = Path.home() / ".filchrc"
        if not config_file.is_file():
                print("ERROR: Configuration file ~/.filchrc does not exist", file=sys.stderr)

        with open(config_file, "rb") as f:
                data = tomllib.load(f)

        database     = data['global']['database']
        url          = data['global']['url']
        ntfy_channel = data['ntfy']['channel']

        args = get_args()

        # Argus Filch
        filch = Filch(args)
        filch.database     = database
        filch.url          = url # URL prefix to the database folder
        filch.ntfy_channel = ntfy_channel

        filch.loop()

if __name__ == "__main__":
        sys.exit(main())
