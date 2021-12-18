import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from random import seed
import random
import sys


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main(video_file):

	# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
	SAVING_FRAMES_PER_SECOND = 1

	cap = cv2.VideoCapture(video_file)

	# get the FPS of the video
	fps = cap.get(cv2.CAP_PROP_FPS)

	# if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
	saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)

	# get the list of duration spots to save
	saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
	    
	# start the loop
	count = 0

	# read in your weights and config file in your model
	yoloweights = "./resource/yolov3.weights"
	yolocfg = "./resource/yolov3.cfg"
	net = cv2.dnn.readNet(yoloweights,yolocfg)


	# load the labels here
	classes = [] ## empty list of python
	file_name = "./resource/label.txt"

	with open(file_name, "rt") as fpt:
	    classes = [line.strip() for line in fpt.readlines()]
	    
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))


	size = (frame_width, frame_height)

	# make a folder by the name of the video file
	filename, _ = os.path.splitext(video_file)
	path_on_system = os.getcwd() # the current working directory

	if not os.path.isdir(filename):
		os.mkdir(filename)
		
	dir_path  = os.path.join(path_on_system, filename)
	os.chdir(dir_path)

        
	# set seed from genearting random number
	seed(1)


	while True:
	        is_read, my_img = cap.read()
	        
	        if not is_read:
	            # break out of the loop if there are no frames to read
	            break
	            
	        # get the duration by dividing the frame count by the FPS
	        frame_duration = count / fps
	        
	        try:
	            # get the earliest duration to save
	            closest_duration = saving_frames_durations[0]
	        except IndexError:
	            # the list is empty, all duration frames were saved
	            break
	            
	         # if closest duration is less than or equals the frame duration, 
	        # then save the frame    
	        if frame_duration >= closest_duration:  
	            ht,wt,_ = my_img.shape
	            blob = cv2.dnn.blobFromImage(my_img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False)
	            net.setInput(blob)
	            last_layer = net.getUnconnectedOutLayersNames()
	            layer_out = net.forward(last_layer) # feed the last layer
	            boxes = []
	            confidences = []
	            class_ids = [] 


	            # need a for loop to figure out the bounding boxes, confidence, and class ids
	            for output in layer_out:
	                for detection in output:
	                    score = detection[5:]   # the probability of each class comes after the 5th element
	                    class_id = np.argmax(score)   # this returns the index of the max prob score
	                    confidence = score[class_id]  # this returns you the confidence of the class id 
	                    if confidence > 0.5:           # only consider the detection if confidence > 0.6
	                        center_x = int(detection[0] * wt)
	                        center_y = int(detection[1] * ht)
	                        w = int(detection[2] * wt)   # this converts back to the normal image size
	                        h = int(detection[3] * ht)

	                        x = int(center_x - w/2)
	                        y = int(center_y - h/2)

	                        boxes.append([x,y,w,h])
	                        confidences.append((float(confidence)))
	                        class_ids.append(class_id)

	            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)   # figure out the bounding boxes
	            font = cv2.FONT_HERSHEY_PLAIN
	            colors = np.random.uniform(0,255, size = (len(boxes),3))

	            for i in indexes.flatten():
	                x,y,w,h = boxes[i]
	                label = str(classes[class_ids[i]])
	                confidence = str(round(confidences[i],2))
	                color = colors[i]
	                if label == "person": # crop the image at the beginning and evert 10 sec
	                    try:
	                        curr_time = time.time()
	                        crop_image = my_img[y-100:y+h+400, x-100:x+w+400]
	                        cv2.imwrite(str(int(curr_time)+random.randint(0,1000))+".jpeg",crop_image) #name the image using time and the index
	                    except:
	                        pass
	                        
	            # drop the duration spot from the list, since this duration spot is already saved
	            try:
	                saving_frames_durations.pop(0)
	            except IndexError:
	                pass

	        count+=1

	        
	print("Finished Cropping the person in video")


if __name__ == "__main__":
    video_file = sys.argv[1]
    main(video_file)
	    
