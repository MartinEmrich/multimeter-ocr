#!/usr/bin/python3


import datetime
import logging as log
print ("Hello")
import matplotlib.pyplot as plt
import sys

log.basicConfig(level=log.DEBUG)

INFILENAME = sys.argv[1]
OUTFILENAME = "powerbank"

f = open(INFILENAME, "r")


print("Wurst")
start = None
timestamps = []
amps = []
volts = []
watthours = []
totalTime = 0
maxMilliAmps = 0.0
maxMilliAmpsTS = None

while True:
    line = f.readline()
    if line is None or not line:
        break;
    #print("Line: {}".format(line))
    tokens = line.split(";")
    ts = datetime.datetime.strptime(tokens[0],"%Y-%m-%d %H:%M:%S")
    p = float(tokens[5])
    u = float(tokens[6])
    i = float(tokens[7])
    E = float(tokens[4])
    if (start is None) and (p > 0.0):
        start = ts
        log.info("Found start timestamp: {}".format(start))

    if start is not None:
        dt = (ts - start).total_seconds()
        totalTime = dt
        if (i <= 0.0):
            totalTime = dt
            break
        timestamps.append( dt / 60.0 )
        amps.append( i)
        if i > maxMilliAmps:
            maxMilliAmps = i
            maxMilliAmpsTS = dt

        volts.append( u )
        watthours.append(E)

maxMilliAmps = maxMilliAmps * 1000.0

log.info("Got {} values".format(len(amps)))
log.info("Max {} mA at {} %".format(maxMilliAmps, (maxMilliAmpsTS / totalTime)*100.0 ))
plt.figure(figsize=(12.0,8.0), dpi=300)
fig, ax = plt.subplots(constrained_layout = True)

#plt.xlabel("Time (s)")
#plt.ylabel("A")
ax.set_xlabel("Time (min), {}:{} min total".format(int(totalTime/60), int(totalTime%60) ))
ax.plot(timestamps, amps, label="Current ({}mA max)".format(maxMilliAmps), color='tab:red')

ax = ax.twinx()
#plt.ylabel("V")
#plt.plot(timestamps, volts)
ax.plot(timestamps, watthours, label="Energy (Wh)", color='tab:blue')

fig.tight_layout()

plt.savefig(OUTFILENAME, dpi=300)

f.close()

