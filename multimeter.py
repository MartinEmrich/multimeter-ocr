#!/usr/bin/python3

# Inspired by: https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

from os import linesep
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import logging as log
import sys
# import math
from pprint import pformat
import numpy as np
from numpy.core.fromnumeric import mean
import pprint

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
DIGIT_VH = [True, False, False, True, False, False, True]

# threshold before edge detection of display window
PRE_EDGE_THRESHOLD = 135

# threshold of mean for one single segment
SEGMENT_THRESHOLD_UPPER = 165
SEGMENT_THRESHOLD_LOWER = 140

ALPHA = 1.50
BETA = 60

NB_RADIUS = 20

# shift of segment locations
GLOBAL_X_OFFSET = 0
GLOBAL_Y_OFFSET = 0

captureDevice = cv2.VideoCapture(videoDev)
log.debug("Frame size")
captureDevice.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
captureDevice.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
log.debug("Exposure")
captureDevice.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0/3
captureDevice.set(cv2.CAP_PROP_EXPOSURE, 600)  # 3-2047
log.debug("Focus")
captureDevice.set(cv2.CAP_PROP_AUTOFOCUS, 0)
captureDevice.set(cv2.CAP_PROP_FOCUS, 0.123)
captureDevice.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)
log.debug("Gain")
captureDevice.set(cv2.CAP_PROP_GAIN, 20)  # 0-255
log.info("FPS: {}".format(int(captureDevice.get(cv2.CAP_PROP_FPS))))

log.debug("Focus mode: {}".format(captureDevice.get(cv2.CAP_PROP_AUTOFOCUS)))
log.debug("Focus value: {}".format(captureDevice.get(cv2.CAP_PROP_FOCUS)))

# 7 segment digit positions
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# Segment center positions

# Segment form
X_STROKE_WIDTH = 24
X_STROKE_LENGTH = 54
Y_STROKE_WIDTH = 30
Y_STROKE_LENGTH = 70


def makeDigit(x, y):
    digit = []
    digit.append([x+64, y+11])

    digit.append([x+28, y+60])
    digit.append([x+98, y+60])

    digit.append([x+54, y+111])

    digit.append([x+17, y+160])
    digit.append([x+86, y+160])

    digit.append([x+44, y+208])
    return digit


DIGIT_POS = []
DIGIT_POS.append(makeDigit(10, 24))
DIGIT_POS.append(makeDigit(120, 24))
DIGIT_POS.append(makeDigit(230, 24))
DIGIT_POS.append(makeDigit(345, 24))

KNOB = 2.0
KD = 0.5

infoImage = None


def captureFrame():
    global captureDevice

    # seems to need a special invitation every time....
    captureDevice.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)
    ret, frame = captureDevice.read()
    return frame


def si(img, name="image"):
    global KNOB
    global ALPHA
    global BETA
    global NB_RADIUS
    global PRE_EDGE_THRESHOLD
    global CAMERA_FOCUS
    cv2.imshow(name, img)
    rv = cv2.waitKey(50)
    if rv == 27:
        sys.exit(0)
    if chr(rv % 256) == 'q':
        KNOB = KNOB - KD
        log.info("KNOB is now {}".format(KNOB))
    if chr(rv % 256) == 'w':
        KNOB = KNOB + KD
    if chr(rv % 256) == 'e':
        NB_RADIUS = NB_RADIUS - 1
        log.info("NB_RADIUS is now {}".format(NB_RADIUS))
    if chr(rv % 256) == 'r':
        NB_RADIUS = NB_RADIUS + 1
        log.info("NB_RADIUS is now {}".format(NB_RADIUS))
    if chr(rv % 256) == 't':
        PRE_EDGE_THRESHOLD = PRE_EDGE_THRESHOLD - 5
        log.info("PRE_EDGE_THRESHOLD is now {}".format(PRE_EDGE_THRESHOLD))
    if chr(rv % 256) == 'z':
        PRE_EDGE_THRESHOLD = PRE_EDGE_THRESHOLD + 5
        log.info("PRE_EDGE_THRESHOLD is now {}".format(PRE_EDGE_THRESHOLD))
    if chr(rv % 256) == 'a':
        CAMERA_FOCUS = CAMERA_FOCUS - 0.05
        log.info("CAMERA_FOCUS is now {}".format(CAMERA_FOCUS))
        captureDevice.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)
    if chr(rv % 256) == 's':
        CAMERA_FOCUS = CAMERA_FOCUS + 0.05
        log.info("CAMERA_FOCUS is now {}".format(CAMERA_FOCUS))
        captureDevice.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)
    if chr(rv % 256) == 'd':
        ALPHA = ALPHA - 0.05
        log.info("ALPHA is now {}".format(ALPHA))
    if chr(rv % 256) == 'f':
        ALPHA = ALPHA + 0.05
        log.info("ALPHA is now {}".format(ALPHA))
    if chr(rv % 256) == 'g':
        BETA = BETA - 5
        log.info("BETA is now {}".format(BETA))
    if chr(rv % 256) == 'h':
        BETA = BETA + 5
        log.info("BETA is now {}".format(BETA))


def freqFilter(image, radius):
    ks = 2*radius+1
    superblurred = cv2.GaussianBlur(image, (ks, ks), 0)

    # smoothed = (image/2.0 - superblurred/2.0)
    smoothed = cv2.subtract(image, superblurred)
    print("smoothed:")
    print(smoothed)
    return smoothed
    # return cv2.normalize(smoothed, 0, 255, cv2.NORM_MINMAX)

# print coordinates clicked on, to help build DIGIT_POS


def onClick(event, x, y, flags, pram):
    if(event == cv2.EVENT_LBUTTONDOWN):
        # log.info("Clicked at [ {}, {} ]".format(x,y))
        print("[{}, {}],".format(x, y))


def preprocessImage(image):
    global infoImage
    global PRE_EDGE_THRESHOLD
    image = imutils.resize(image, width=640)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.threshold(blurred, PRE_EDGE_THRESHOLD,
                          255, cv2.THRESH_BINARY)[1]

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

    numContours = len(contours)

    #log.debug("got {} contours".format(numContours))
    displayCnt = None
    # loop over the contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        if (peri < 1000.0):  # too small, definitely an error
            continue

        epsilon = peri * 0.02
        approx = cv2.approxPolyDP(c, epsilon, True)

        # it's a rectangle
        if len(approx) == 4:
            displayCnt = approx
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            break

    if displayCnt is None:
        log.warn("Found no display frame")
        
        if numContours > 10: # noisy?
            log.info("trying to reduce pre edge threshold")
            PRE_EDGE_THRESHOLD = PRE_EDGE_THRESHOLD - 1
        else:
            log.info("trying to increase pre edge threshold")
            PRE_EDGE_THRESHOLD = PRE_EDGE_THRESHOLD + 1

        return None

    # extract the display, apply a perspective transform
    # to it
    try:
        gray = four_point_transform(gray, displayCnt.reshape(4, 2))
        # infoImage = four_point_transform(image, displayCnt.reshape(4, 2))
    except AttributeError:
        log.warn("Apparently no edges found, transform failed.")
        return None

    # here gray seems to be uint8
    #gray = cv2.convertScaleAbs(gray, alpha=ALPHA, beta=BETA)

    si(gray, "scaled")

    #gray = cv2.GaussianBlur(gray, (9, 9), 0)

    #si(gray, "pre-blurred")

    # ... blocksize, threshold-change-constant)
    digitBW = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, NB_RADIUS*2+1, KNOB)  # [1]

    infoImage = cv2.cvtColor(digitBW, cv2.COLOR_GRAY2RGB)

    cv2.namedWindow("final: digits")
    cv2.setMouseCallback("final: digits", onClick)
    cv2.imshow("final: digits", digitBW)

    return digitBW


def analyzeDigit(image, segmentset):
    global infoImage
    index = 0
    result = []
    for segment in segmentset:
        direction = DIGIT_VH[index]

        if direction:
            x1offset = -X_STROKE_LENGTH/2
            x2offset = X_STROKE_LENGTH/2
            y1offset = -X_STROKE_WIDTH/2
            y2offset = X_STROKE_WIDTH/2
        else:
            x1offset = -Y_STROKE_WIDTH/2
            x2offset = Y_STROKE_WIDTH/2
            y1offset = -Y_STROKE_LENGTH/2
            y2offset = Y_STROKE_LENGTH/2

        x1 = int(segment[0]+x1offset) + GLOBAL_X_OFFSET
        y1 = int(segment[1]+y1offset) + GLOBAL_Y_OFFSET

        x2 = int(segment[0]+x2offset) + GLOBAL_X_OFFSET
        y2 = int(segment[1]+y2offset) + GLOBAL_Y_OFFSET
        # log.debug("segment block at {},{} - {},{}".format(x1,y1,x2,y2))

        # Watch out: x and y are flipped in numpy
        cutout = image[y1:y2, x1:x2]

        meanVal = cutout.mean(axis=0).mean(axis=0)
        if (meanVal < SEGMENT_THRESHOLD_LOWER):
            # log.debug("Segment is active (black)")
            segmentValue = 1
            infoImage = cv2.rectangle(
                infoImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif(meanVal > SEGMENT_THRESHOLD_UPPER):
            # log.debug("Segment is inactive (white/gray)")
            segmentValue = 0
            infoImage = cv2.rectangle(
                infoImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            infoImage = cv2.rectangle(
                infoImage, (x1, y1), (x2, y2), (255, 128, 128), 2)
            segmentValue = None

        infoImage = cv2.putText(infoImage, "{:0.1f}".format(
            meanVal), (x1, y2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 160), 2)

        result.append(segmentValue)
        index = index + 1

    return result


def analyzeDigits(image):
    result = []
    success = True
    for segmentset in DIGIT_POS:
        # segmentset = DIGIT_POS[0]
        digit_bits = tuple(analyzeDigit(image, segmentset))

        if digit_bits in DIGITS_LOOKUP:
            digit = DIGITS_LOOKUP[digit_bits]
        else:
            digit = "?"
            success = False

        result.append(digit)
    return result, success


def getMultimeterValue():
    image = captureFrame()
    if (image is None):
        log.error("Failed to acquire image")
        sys.exit(1)

    si(image, "raw")

    image = preprocessImage(image)

    if image is not None:
        result, success = analyzeDigits(image)
        si(infoImage, "result")

        if (success):
            # some AI: values > 3.0 are improbable, probably misread 1 for a 7
            fixed_first_digit = result[0]
            if result[0] == 7:
                fixed_first_digit = 1

            amps = fixed_first_digit + \
                result[1]*0.1 + result[2]*0.01 + result[3]*0.001
            # log.info("Current: {}mA".format(1000.0 * amps))
            return amps
        else:
            log.warn("Incomplete result: {}".format(result))
            return None
