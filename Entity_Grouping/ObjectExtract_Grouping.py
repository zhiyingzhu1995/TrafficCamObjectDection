
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
from random import seed
import random


def main(video_file):
	cap = cv2.VideoCapture(video_file)

	yoloweights = "./resource/yolov3.weights"
	yolocfg = "./resource/yolov3.cfg"

	# read in your weights and config file in your model
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
	path_on_system = os.getcwd() # the current working directory
	filename = "ProcessedVideo"

	if not os.path.isdir(filename):
		os.mkdir(filename)
		
	dir_path  = os.path.join(path_on_system, filename)
	os.chdir(dir_path)



	# the resulting video
	threshold = 0.5
	result = cv2.VideoWriter(f"Processed_Video_Asbury_Park_{threshold}.avi", 
	                         cv2.VideoWriter_fourcc(*'MJPG'),
	                         10, size)

	# do a while loop to do this on each image frame
	while True:
	    is_read, my_img = cap.read()
	        
        if not is_read:
            # break out of the loop if there are no frames to read
            break
	            
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
	            if confidence > threshold:           # only consider the detection if confidence > 0.6
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

	    # create a dictionary for index and boxes x-component
	    ind_box = {}

	    for ind in indexes.flatten():
	        ind_box[boxes[ind][0]] = ind

	    # sort the box and index by boxes x-components
	    ind_box = dict(sorted(ind_box.items())) # {226: 12} = Box-x: index

	    x_prior, y_prior, w_prior, h_prior = 0,0,0,0
	    label_prior = ""
	    CI_prior = 0
	    indexes_new = []


	    for k,i in ind_box.items():
	        intersection_area = 0
	        x,y,w,h = boxes[i]
	        label = str(classes[class_ids[i]])
	        confidence = str(round(confidences[i],2))


	        # calculate the if intersection happens
	        x_left = max(x, x_prior)
	        x_right = min(x+w,x_prior+w_prior)
	        y_bottom = min(y,y_prior)
	        y_top = max(y+h,y_prior+h_prior)

	        if x_right > x_left or y_bottom > y_top:
	            intersection_area = abs((x_right - x_left) * (y_bottom - y_top))


	        # change names to biker
	        if ((label_prior == "person" and label == "bicycle") or (label_prior == "bicycle" and label == "person")) and intersection_area > 0:
	            # pop the previous detected object
	            indexes_new.pop()
	            indexes_new.append(i)
	            # change the current label to a new label
	            class_ids[i] = -2        # index -2 is biker
	            # change the prior label to current new label
	            label_new = "biker"
	            # change the boxes
	            x_new = min(x,x_prior)
	            y_new = min(y,y_prior)
	            w_new = max(x+w,x_prior+w_prior)-min(x,x_prior)
	            h_new = max(y+h,y_prior+h_prior)-min(y,y_prior)
	            boxes[i] = [x_new,y_new,w_new,h_new]
	            # change the max CL to max 
	            confidences[i] = max(CI_prior,round(confidences[i],2))
	            # set the prior label to the current label
	            label_prior = label_new
	            x_prior, y_prior, w_prior, h_prior = x,y,w,h
	            CI_prior = round(confidences[i],2)

	        # change names to scooter
	        if ((label_prior == "person" and label == "skateboard") or (label_prior == "skateboard" and label == "person")) and intersection_area > 0:
	            # pop the previous detected object
	            indexes_new.pop()
	            indexes_new.append(i)
	            # change the current label to a new label
	            class_ids[i] = -1
	            # change the prior label to current new label
	            label_new = "scooter"
	            # change the boxes
	            x_new = min(x,x_prior)
	            y_new = min(y,y_prior)
	            w_new = max(x+w,x_prior+w_prior)-min(x,x_prior)
	            h_new = max(y+h,y_prior+h_prior)-min(y,y_prior)
	            boxes[i] = [x_new,y_new,w_new,h_new]
	            # change the max CL to max 
	            confidences[i] = max(CI_prior,round(confidences[i],2))
	            # set the prior label to the current label
	            label_prior = label_new
	            x_prior, y_prior, w_prior, h_prior = x,y,w,h
	            CI_prior = round(confidences[i],2)

	        else:
	            # set the prior label to the current label
	            indexes_new.append(i)
	            label_prior = label
	            x_prior, y_prior, w_prior, h_prior = x,y,w,h
	            CI_prior = round(confidences[i],2)
	     
	            
	    
	    # write a for loop to draw all the boxes
	    for i in indexes_new:
	        x,y,w,h = boxes[i]
	        label = str(classes[class_ids[i]])
	        confidence = str(round(confidences[i],2))
	        color = colors[i]
	        cv2.rectangle(my_img, (x,y), (x+w,y+h), color,2) # draw the bounding box
	        cv2.putText(my_img, label + "" + confidence, (x,y+20), font, 2, (0,0,255),2)
	    
	    #cv2.imshow("boxed",my_img)
	    result.write(my_img)

	    
	result.release()    
	print("The video was successfully saved")


if __name__ == "__main__":
    video_file = sys.argv[1]
    main(video_file)
	    
