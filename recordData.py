#!/usr/bin/python3

import pprint
import vxi11
import sys
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
import re
import multimeter
import logging as log
import threading

log.basicConfig(level=log.DEBUG)
log.getLogger("apscheduler.scheduler").setLevel(level=log.WARN)
log.getLogger("apscheduler.executors.default").setLevel(level=log.WARN)

class siglentpsu:
    inst = vxi11.Instrument("192.168.178.118")
    ch = "CH2"

    def getPower(self):
        self.inst.write("MEAS:POWE? " + ch)
        return float(self.inst.read())

    def getVolt(self):
        self.inst.write("MEAS:VOLT? " + ch)
        return float(self.inst.read())

    def getCurrent(self):
        self.inst.write("MEAS:CURR? " + ch)
        return float(self.inst.read())


class siglentscope:
    inst = vxi11.Instrument("192.168.178.120")
    voltParser = re.compile("C([0-9]):PAVA MEAN,([0-9.]+)E([0-9+-]+)V")

    def getVoltsFromScope(self):
        self.inst.write("C1:PAVA? MEAN")
        rv = self.inst.read()
        match = self.voltParser.search(rv)
        if match is not None:
            channel = int(match.groups()[0])
            mant = float(match.groups()[1])
            expo = int(match.groups()[2])
            volts = mant * pow(10, expo)
            return (volts, channel)
        return (None, None)


class AmpUpdater:
    lastAmps = None
    thread = None

    def __init__(self, interval=1):
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def updateAmps(self):
        i = multimeter.getMultimeterValue()
        if i is not None:
            self.lastAmps = i
            return i
        log.warn("Failed to get amps from multimeter, reusing old value {}".format(
            self.lastAmps))

    def run(self):
        while(True):
            self.updateAmps()

    def get(self):
        return self.lastAmps


scope = siglentscope()

(voltage, channel) = scope.getVoltsFromScope()
print("Got {}V".format(voltage))


totalenergy = 0.0

datafile = open('log.txt', 'a')


startTime = datetime.datetime.now()
last = startTime
energy = 0
ampUpdate = AmpUpdater()


def processTick():
    global ampUpdate
    global totalenergy
    global last

    now = datetime.datetime.now()

    (volt, channel) = scope.getVoltsFromScope()
    current = ampUpdate.get()
    if current is None:
        log.warning("no valid value from multimeter yet, skipping")
        return

    power = volt * current

    timediff = now - last
    tickSeconds = timediff.total_seconds()

    last = now

    energyWattSeconds = power * tickSeconds
    energy = energyWattSeconds / 3600.0
    totalenergy = totalenergy + energy

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print("{} tick: {:5.2f} seconds, {: >3.3f} Ws ({:3.5f} mWh), {:2.6f}Wh total, currently {:1.3f}W {:1.3f}V {:1.3f}A".format(
        timestamp, tickSeconds, energyWattSeconds, energy, totalenergy, power, volt, current))
    print("{};{};{};{};{};{};{};{}".format(timestamp, tickSeconds,
                                           energyWattSeconds, energy, totalenergy, power, volt, current), file=datafile)


sched = BlockingScheduler()

sched.add_job(processTick, 'interval', seconds=2)


sched.start()
