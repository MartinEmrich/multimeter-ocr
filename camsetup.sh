#!/bin/bash

#v4l2-ctl -d /dev/video0 --list-ctrls

v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=90