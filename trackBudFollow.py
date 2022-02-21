#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse
import sys
import time 

sys.path.insert(0, '~/jetbot/jetbot') #Hopefully lets script find module for motor control

from jetbot import Robot
trackBud = Robot()

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--kp", type=float, default=0.1, help="proportional gain for tracking controller")
parser.add_argument("--kd", type=float, default=0.01, help="derivative gain for tracking controller")
parser.add_argument("--maxspeed", type=float, default=0.8, help="max wheelspeed")
parser.add_argument("--minspeed", type=float, default=0.2, help="min wheelspeed")
parser.add_argument("--dt", type=float, default=500, help="sampling rate for controller (ms)") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
	
# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

# Run Control Loop
dt = opt.dt #sec
Kp = opt.kp
Kd = opt.kd
last_time = time.perf_counter()*1000
last_error = 0
high_wspeed_bound = opt.maxspeed
low_wspeed_bound = opt.minspeed
base_speed = (high_wspeed_bound - low_wspeed_bound)/2 #base speed for forward motion
while True:
    current_time = time.perf_counter()*1000
    if current_time - last_time >= dt:
        last_time = current_time
        # capture the next image
        img = input.Capture()

        # detect objects in the image (with overlay)
        detections = net.Detect(img, overlay=opt.overlay)

        # print the detections
        print("detected {:d} objects in image".format(len(detections)))

        people = [det for det in detections if det.ClassID == 1] #list of people objects in frame
        people.sort(key = lambda x: img.width/2 - x.Center[0]) #sort by horizontal proximity to center of frame, done for convenience because we one-nighted this mofo

        # create error signal, horz.distance of closest person in bounding box to center of frame
        error = people[0].center[0] - img.width/2
        K = Kp*error + Kd*(error - last_error)/(dt/1000) #Control signal, corresponds to magnitude of difference in wheelspeeds
        print("gain: ",K," error (pixels): ", error)
        last_error = error

        #Set wheelspeeds accordingly
        trackBud.left_motor.value = sorted((low_wspeed_bound,0.5 + K,high_wspeed_bound))[1]
        trackBud.right_motor.value = sorted((low_wspeed_bound,0.5 - K,high_wspeed_bound))[1]     

	# exit on input EOS
	if not input.IsStreaming():
		break
