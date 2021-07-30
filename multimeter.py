#!/usr/bin/python3

# Source, Inspiration: https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

from os import linesep
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import logging as log
import sys
import math
from pprint import pformat
import numpy as np
from numpy.core.fromnumeric import mean

log.basicConfig(level=log.DEBUG)


########################################
# Settings
########################################

# Video device number
videoDev = 0

# Camera autofocus value (0.0-1.0)
CAMERA_FOCUS = 0.35

# Segment orientation
# True: Left-Right, False: Up-Down
DIGIT_VH = [ True, False, False, True, False, False, True ]

# Segment form
STROKE_WIDTH = 20
STROKE_LENGTH = 50

# threshold for whole display cutout
CUTOUT_THRESHOLD = 0.45

# threshold of mean for one single segment
SEGMENT_THRESHOLD = 0.2


captureDevice = cv2.VideoCapture(videoDev)
log.debug("Frame size")
captureDevice.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
captureDevice.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
log.debug("Exposure")
captureDevice.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0/3
captureDevice.set(cv2.CAP_PROP_EXPOSURE, 600)  # 3-2047
log.debug("Focus")
captureDevice.set(cv2.CAP_PROP_AUTOFOCUS, 0)
captureDevice.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)
log.debug("Gain")
captureDevice.set(cv2.CAP_PROP_GAIN, 20)  # 0-255
log.info("FPS: {}".format(int(captureDevice.get(cv2.CAP_PROP_FPS))))

log.debug("Focus mode: {}".format(captureDevice.get(cv2.CAP_PROP_AUTOFOCUS)))
log.debug("Focus value: {}".format(captureDevice.get(cv2.CAP_PROP_FOCUS)))

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# Segment center positions
DIGIT_POS = [
    [
        [78, 37],
        [34, 89],
        [97, 90],
        [60, 128],
        [26, 177],
        [91, 181],
        [59, 228]
    ], [
        [179, 42],
        [144, 78],
        [209, 81],
        [167, 134],
        [135, 183],
        [199, 185],
        [159, 232]
    ], [
        [289, 44],
        [252, 85],
        [318, 82],
        [281, 138],
        [244, 184],
        [308, 186],
        [266, 235]
    ], [
        [399, 42],
        [365, 83],
        [422, 90],
        [390, 137],
        [350, 182],
        [415, 189],
        [376, 233]
    ]
]


KNOB = 0.5
KD = 0.05

infoImage = None

def captureFrame():
    global captureDevice
    ret, frame = captureDevice.read()
    return frame


def si(img, name="image"):
    global KNOB
    cv2.imshow(name, img)
    rv = cv2.waitKey(50)
    if rv == 27:
        sys.exit(0)
    if chr(rv % 256) == 'q':
        KNOB = KNOB - KD
        log.info("KNOB is now {}".format(KNOB))
    if chr(rv % 256) == 'w':
        KNOB = KNOB + KD
        log.info("KNOB is now {}".format(KNOB))


def freqFilter(image, radius):
    ks = 2*radius+1
    superblurred = cv2.GaussianBlur(image, (ks, ks), 0)
    return cv2.normalize((image/2.0 - superblurred/2.0), 0, 255, cv2.NORM_MINMAX)


def onClick(event, x, y, flags, pram):
    if(event == cv2.EVENT_LBUTTONDOWN):
        #log.info("Clicked at [ {}, {} ]".format(x,y))
        print("[{}, {}],".format(x, y))


def preprocessImage(image):
    global infoImage
    image = imutils.resize(image, width=640)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

    # Canny: Threshold1, Threshold2, aperturesize
    edged = cv2.Canny(edged, 30, 80, 255)

    si(edged, "preview")

    #
    # Find display border contours
    #

    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=False)

    #log.debug("got {} contours".format(len(contours)))
    displayCnt = None
    # loop over the contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        epsilon = peri * 0.02
        approx = cv2.approxPolyDP(c, epsilon, True)
        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            break

    if displayCnt is None:
        log.warn("Found no display frame.")
        return None

    # extract the display, apply a perspective transform
    # to it
    try:
        gray = four_point_transform(gray, displayCnt.reshape(4, 2))
        infoImage = four_point_transform(image, displayCnt.reshape(4, 2))
    except AttributeError:
        log.warn("Apparently no edges found, transform failed.")
        return None

    filtered = freqFilter(gray, 60)

    si(filtered, "filtered")

    digitBW = cv2.threshold(filtered, CUTOUT_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    si(digitBW, "digits")

    cv2.namedWindow("digits")
    cv2.setMouseCallback("digits", onClick)
    cv2.imshow("digits", digitBW)

    return filtered

def analyzeDigit(image, segmentset):
    global infoImage
    global SEGMENT_THRESHOLD
    index = 0
    result = []
    for segment in segmentset:
        direction = DIGIT_VH[index]

        if direction:
            x1offset = -STROKE_LENGTH/2
            x2offset = STROKE_LENGTH/2
            y1offset = -STROKE_WIDTH/2
            y2offset = STROKE_WIDTH/2
        else:
            x1offset = -STROKE_WIDTH/2
            x2offset = STROKE_WIDTH/2
            y1offset = -STROKE_LENGTH/2
            y2offset = STROKE_LENGTH/2
        
        x1 = int(segment[0]+x1offset)
        y1 = int(segment[1]+y1offset)

        x2 = int(segment[0]+x2offset)
        y2 = int(segment[1]+y2offset)
        #log.debug("segment block at {},{} - {},{}".format(x1,y1,x2,y2))
        
        # Watch out: x and y are flipped in numpy
        cutout = image[y1:y2, x1:x2]

        meanVal = cutout.mean(axis=0).mean(axis=0)
        #log.debug("Mean {}".format(meanVal))
        if (meanVal < SEGMENT_THRESHOLD):
            #log.debug("Segment is active (black)")
            segmentValue = 1
            infoImage = cv2.rectangle(infoImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            #log.debug("Segment is inactive (white/gray)")
            segmentValue = 0
            infoImage = cv2.rectangle(infoImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
        result.append(segmentValue)
        index = index + 1
    
    return result

def analyzeDigits(image):
    result = []
    success = True
    for segmentset in DIGIT_POS:        
    #segmentset = DIGIT_POS[0]
        digit_bits = tuple(analyzeDigit(image, segmentset))
        
        if digit_bits in DIGITS_LOOKUP:
            digit = DIGITS_LOOKUP[digit_bits]
        else:
            log.warn("Failed to decode digit.")
            digit = "?"
            success = False

        result.append(digit)
    return result, success

while True:
    #image = cv2.imread("/home/martin/Dropbox/2021-07-30-173232.jpg")
    image = captureFrame()
    if (image is None):
        log.error("Failed to acquire image")
        sys.exit(1)

    image = preprocessImage(image)

    if image is not None:
        result, success = analyzeDigits(image)
        si(infoImage, "result")

        if (success):
            amps = result[0] + result[1]*0.1 +  result[2]*0.01 +  result[3]*0.001
            log.info("Current: {}mA".format(1000.0 * amps))
        else:
            log.warn("Incomplete result: {}".format(result))


sys.exit(1)
