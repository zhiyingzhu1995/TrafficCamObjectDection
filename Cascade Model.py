
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
from random import seed
import random


def main(video_file):
    
    # function for creating the neural net work and outputting the last layer
    def neuralNetworkFromImage(my_img, net_ = "YOLO"):
        classes = [] 
        boxes = []
        confidences = []
        class_ids = [] 
        ht,wt,_ = my_img.shape
        
        if net_ == "YOLO":
            # YOLO weights
            yoloweights = "./resource/yolov3.weights"
            yolocfg = "./resource/yolov3.cfg"
            net = cv2.dnn.readNet(yoloweights,yolocfg)
            
            # YOLO Class label
            file_name = "./resource/label.txt"
            
            with open(file_name, "rt") as fpt:
                classes = [line.strip() for line in fpt.readlines()]
            ht,wt,_ = my_img.shape 
            blob = cv2.dnn.blobFromImage(my_img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False) 
            net.setInput(blob)
            last_layer = net.getUnconnectedOutLayersNames()
            layer_out = net.forward(last_layer)

            
        if net_ == "Customize":
            # Customized Label weights
            custweights = "./resource/yolov3_training_last_new.weights"
            custcfg = "./resource/yolov3_testing_new.cfg"
            net = cv2.dnn.readNet(custweights,custcfg)
            # Customized Class Label
            file_name = "label_2_class.txt"
            with open(file_name, "rt") as fpt:
                classes = [line.strip() for line in fpt.readlines()]
            ht,wt,_ = my_img.shape 
            blob = cv2.dnn.blobFromImage(my_img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False) 
            net.setInput(blob)
            last_layer = net.getUnconnectedOutLayersNames()
            layer_out = net.forward(last_layer)
            
        

        # First feed in the YOLOV3 model first, if person is detected, then pass that to the Customized label
        # need a for loop to figure out the bounding boxes, confidence, and class ids
        for output in layer_out:
            for detection in output:
                score = detection[5:]   # the probability of each class comes after the 5th element
                class_id = np.argmax(score)   # this returns the index of the max prob score
                confidence = score[class_id]  # this returns you the confidence of the class id 
                # check in the box and see what color is detected
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)   # figure out the bounding boxes based on confidences
        return indexes,classes,boxes,confidences, class_ids

    def getNewLabel(indexes,classes,boxes,confidences, class_ids):
        # new label for holding
        indexes_new = []

        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])

            # if the label is not person, skateboard, bicycle, add the index to new index
            if label not in ["person", 'skateboard', 'bicycle']:
                indexes_new.append(i)

            # if the label is person, expand the image bounding box and call the neural function again to get the new label
            if label == "person":
                crop_image = my_img[y-100:y+h+400, x-100:x+w+400]
                indexes_,classes_,boxes_,confidences_, class_ids_ = neuralNetworkFromImage(crop_image, net_ = "Customize")
                # if the customized object is not detected, that means it's a person, do not change the original label
                if len(indexes_) == 0:
                    indexes_new.append(i)
                # if the detected new label is scooter, then change the old label index to be -1 to indicate scooter
                if len(indexes_) != 0 and classes_[class_ids_[indexes_.flatten()[0]]] == "Scooter Rider":
                    indexes_new.append(i)
                    class_ids[i] = -1
                    confidences[i] = confidences_[indexes_.flatten()[0]] # change the confidence to the new confidence
                if len(indexes_) != 0 and classes_[class_ids_[indexes_.flatten()[0]]] == "Biker":
                    indexes_new.append(i)
                    class_ids[i] = -2
                    confidences[i] = confidences_[indexes_.flatten()[0]]  # change the confidence to the new confidence
            

        return input


        # Read in video
	cap = cv2.VideoCapture(video_file)
	    
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
	result = cv2.VideoWriter(f"Processed_Video_Asbury_Park_{threshold}_cascade_method.avi", 
	                         cv2.VideoWriter_fourcc(*'MJPG'),
	                         10, size)

	# do a while loop to do this on each image frame
	while True:
	    is_read, my_img = cap.read()
	        
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            
            # get the indexes from the YOLO model
            indexes,classes,boxes,confidences, class_ids = neuralNetworkFromImage(my_img, net_ = "YOLO")
            
            # YOLO with customized label
            indexes_new = getNewLabel(indexes,classes,boxes,confidences, class_ids)


            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0,255, size = (len(boxes),3))

            # write a for loop to draw all the boxes
            # if a label is detected, then draw the box, if no label is detected, dont draw the box
            if len(indexes_new) !=0: 
                for i in indexes_new:
                    x,y,w,h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(my_img, (x,y), (x+w,y+h), color,2) # draw the bounding box
                    cv2.putText(my_img, label + "" + confidence, (x,y+20), font, 3, (0,0,255),4)
                        
	    #cv2.imshow("boxed",my_img)
	    result.write(my_img)

	    
	result.release()    
	print("The video was successfully saved")


if __name__ == "__main__":
    video_file = sys.argv[1]
    main(video_file)
	    
