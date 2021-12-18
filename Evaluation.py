import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from random import seed
import random
import json
from collections import Counter

"""
function for creating the neural net work and outputting the last layer
"""
def neuralNetworkFromImages(my_img):
    classes = [] 
    boxes = []
    confidences = []
    class_ids = [] 
    ht,wt,_ = my_img.shape

    # new customized weights and cfg and class
    custweights = "./resource/yolov3_training_last_new.weights"
    custcfg = "./resource/yolov3_training_new.cfg"
    file_name = "./resource/label_2_class.txt"
    
    net = cv2.dnn.readNet(custweights,custcfg)
    
    
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



#################################################
"""
Evaluation Function
"""
def Evaluate(ImagePath, ImageOutput, currentPath):
    # Get all the biker image
    os.chdir(ImagePath)
    ImageList = []
    LabelChecking = {} # compare the ground truth label to label from the image bounding box with highest overlapping area
    iouDict = {} # calculate the iou for each detected boxes
    gtNLabels = {} # number of labels detected by ground truth
    pdNlabels = {} # number of labels detected by model


    for file in glob.glob("*.json"):
        ImageList.append(ImagePath+ "/" + file)
        
        
    for f in ImageList:
        file = open(f,"r")
        img_name = f.split("/")[-1].replace(".json", ".jpeg")
        imgPath = f.replace(".json", ".jpeg")
        my_img = plt.imread(imgPath)
        coorLabel_GroundTruth = {}
        
        
        for line in file:
            json_string =json.loads(line)
            meta = json_string[0]
            annotation = meta['annotations']

            for items in annotation:
                label = items["label"]
                coordinates = items['coordinates']
                # x, y are centers, need to convert them to the actual bounding box coordinate
                x = coordinates['x']
                y = coordinates ['y']
                w = coordinates ['width']
                h = coordinates ['height']
                # convert them to the actual bounding box coordinates
                x = x - int(w/2)
                y = y - int(h/2)
                # insert into the coordinate label ground truth dictionary
                coordStr = str(int(x)) + "," + str(int(y)) + "," + str(int(w)) + "," + str(int(h))
                coorLabel_GroundTruth[coordStr] = label
                
        #####################################
        ### Begin customized model label detection
        ###
        #########################################
        os.chdir(currentPath)  

        indexes,classes,boxes,confidences, class_ids = neuralNetworkFromImages(my_img)

        # write a for loop to draw all the boxes
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255, size = (len(boxes),3))
        coorLabel_predicted = {}
        
        if len(indexes) != 0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == 'Scooter Rider': label = 'Scooter' # 
                coordStr = str(int(x)) + "," + str(int(y)) + "," + str(int(w)) + "," + str(int(h))
                coorLabel_predicted[coordStr] = label.lower()
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(my_img, (x,y), (x+w,y+h), color,3) # draw the bounding box, cv2.rectangle(image, start_point, end_point, color, thickness)
                cv2.putText(my_img, label + "" + confidence, (x,y+20), font, 3, (0,0,255),4) # v2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])

        os.chdir(ImageOutput)
        cv2.imwrite('annotated_'+img_name,my_img)


        #########################################
        ### Do pairwise comparison to get the bounding boxes
        ### with the highest Intersecting Area
        #########################################  
        img_name = img_name.replace(".jpeg", "")
    
        for k,v in coorLabel_GroundTruth.items():
            x = int(k.split(",")[0])
            y = int(k.split(",")[1])
            w = int(k.split(",")[2])
            h = int(k.split(",")[3])
            label = v
            intersectionList = []
            iouList = []

            # if there is detected labels, then calculate the intersecting area:
            if len(coorLabel_predicted.items()) != 0:
                for k_pre, v_pre in coorLabel_predicted.items():
                    # assum intersection area is 0
                    intersection_area = 0

                    x_pre = int(k_pre.split(",")[0])
                    y_pre = int(k_pre.split(",")[1])
                    w_pre = int(k_pre.split(",")[2])
                    h_pre = int(k_pre.split(",")[3])
                    label_pre = v_pre

                    # calculate the if intersection happens
                    x_left = max(x, x_pre)
                    x_right = min(x+w,x_pre+w_pre)
                    y_bottom = min(y,y_pre)
                    y_top = max(y+h,y_pre+h_pre)

                    # update the intersection area accordingly
                    if x_right > x_left or y_bottom > y_top:
                        intersection_area = abs((x_right - x_left) * (y_bottom - y_top))

                    # calculate the area of union for IOU = area of overlap / area of union
                    gtbox = abs(w * h)
                    pdbox = abs(w_pre * h_pre)
                    union_area = gtbox + pdbox - intersection_area
                    iou = round(intersection_area/union_area,3)

                        

                    iouList.append(iou)
                    intersectionList.append(intersection_area)

                # Find the highest overlapping areas for each of the ground truth label
                IndexForHighestArea = np.argmax(intersectionList)
                # Find the corresponding label and coordinates from the highest overlapping areas from predicted image
                HigestImageCoord = list(coorLabel_predicted.keys())[IndexForHighestArea]
                HigestImageLabel = list(coorLabel_predicted.values())[IndexForHighestArea]
                # Check if the label matches with the ground truth label and store that as the result
                result = HigestImageLabel == label
                LabelChecking[img_name + "_" + k] = result
                # store the calcualte iou for the predicted box with highest overlapping area
                HigestIoU = iouList[IndexForHighestArea]
                iouDict[img_name + "_" + k] = HigestIoU

            # if there is no predicted labels, then record no label predicted, and na
            else:
                LabelChecking[img_name + "_" + k] = "Not Label Predicted"
                iouDict[img_name + "_" + k] = "NA"

            # store the number of labels from grouth true
            gtNLabels [img_name] = len(coorLabel_GroundTruth)
            pdNlabels [img_name] = len(coorLabel_predicted)
        
    
        file.close()
        
    return LabelChecking, iouDict, gtNLabels, pdNlabels


def main():

    ########
    #Biker Image Eval
    ########
    
    # Biker Testing image path
    bikerImagePath = <'fill in your path'>
    bikerImageOutput = <'fill in your path'>

    # current path
    currentPath = <'fill in your path'>

    
    # Evaluate on Biker images 
    LabelCheckingBiker, iouDictBiker, gtNLabelsBiker, pdNlabelsBiker = Evaluate(bikerImagePath,bikerImageOutput, currentPath )


    ########
    #Scooter Image Eval
    ########

    scooterImagePath  =  <'fill in your path'>
    scooterImageOutput = <'fill in your path'>

    currentPath = <'fill in your path'>

    LabelCheckingScooter, iouDictScooter, gtNLabelsScooter, pdNlabelsScooter = Evaluate(scooterImagePath,scooterImageOutput, currentPath )

    ########
    #Analysis
    ########
    
    #print the number of images
    TotalImageBiker = len(gtNLabelsBiker)
    #print the number of total ground truth bounding boxes
    TotalBxBiker = len(LabelCheckingBiker)
    NumTrueBiker  = dict(Counter(list(LabelCheckingBiker.values())))[True]
    NumFalseBiker = dict(Counter(list(LabelCheckingBiker.values())))[False]
    NumMissBiker = dict(Counter(list(LabelCheckingBiker.values())))['Not Label Predicted']
    AccuracyBiker = round(NumTrueBiker / (NumTrueBiker+NumFalseBiker+NumMissBiker) * 100, 2)
    iouNonNaBiker = [item for item in list(iouDictBiker.values()) if item !='NA'] 
    averageIOUBiker = round(sum(iouNonNaBiker)/len(iouNonNaBiker)*100, 2)


    
    #print the number of images
    TotalImageScooter = len(gtNLabelsScooter)
    #print the number of total ground truth bounding boxes
    TotalBxScooter = len(LabelCheckingScooter)
    NumTrueScooter  = dict(Counter(list(LabelCheckingScooter.values())))[True]
    NumFalseScooter = dict(Counter(list(LabelCheckingScooter.values())))[False]
    NumMissScooter = dict(Counter(list(LabelCheckingScooter.values())))['Not Label Predicted']
    AccuracyScooter = round(NumTrueScooter / (NumTrueScooter+NumFalseScooter+NumMissScooter) * 100, 2)
    iouNonNaScooter = [item for item in list(iouDictScooter.values()) if item !='NA' and item != 0 and abs(item) < 50] 
    averageIOUScooter = round(sum(iouNonNaScooter)/len(iouNonNaScooter)*100, 2)


    #print the number of images
    TotalImage = TotalImageScooter + TotalImageBiker
    print("Total number of scooter and biker images: ", TotalImage)

    #print the number of total ground truth bounding boxes
    TotalBx = TotalBxScooter + TotalBxBiker
    print("Total number of ground truth bounding boxes is {}".format(TotalBx))
    NumTrueTotal = NumTrueBiker + NumTrueScooter
    NumFalseTotal = NumFalseBiker + NumFalseScooter
    NumMissTotal = NumMissBiker + NumMissScooter
    AccuracyTotal = round(NumTrueTotal / (NumTrueTotal+NumFalseTotal+NumMissTotal) * 100, 2)
    averageIOUTotal = round((sum(iouNonNaScooter) + sum(iouNonNaBiker))/ (len(iouNonNaScooter) +len(iouNonNaBiker))  *100, 2)


    Columns = ['Biker', 'Scooter', 'Biker+Scooter']
    Index = ['Num of Img  ', 'Num of Box  ','Num of True  ', 'Num of False  ', 'Num of Miss  ', 
             'Accuracy (%)  ', 'Avg IOU (%)  ']
    TotalNumImage  =  [TotalImageBiker, TotalImageScooter,TotalImage] 
    TotalNumBx = [TotalBxBiker,TotalBxScooter,TotalBx ]
    TotalTrue = [NumTrueBiker,NumTrueScooter,NumTrueTotal]
    TotalFalse = [NumFalseBiker,NumFalseScooter,NumFalseTotal]
    TotalMiss = [NumMissBiker,NumMissScooter,NumMissTotal]
    TotalAccuracy = [AccuracyBiker,AccuracyScooter,AccuracyTotal]
    TotalAvgIou = [averageIOUBiker, averageIOUScooter, averageIOUTotal]

    df = pd.DataFrame()

    df = pd.DataFrame(np.column_stack(list(zip(TotalNumImage, TotalNumBx, TotalTrue, TotalFalse,TotalMiss,  TotalAccuracy,TotalAvgIou))),
                  columns = Columns,
                  index = Index
                 )
    
    df.to_csv('eval_result.csv', index=False)

if __name__ == "__main__":
    main()
	    

