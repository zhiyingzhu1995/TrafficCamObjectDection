# TrafficCamObjectDection

## Design:
    I. Data Collection
To understand the real traffic situation, a camera is replaced to record the 24-hour traffic interactions at Asbury Ave, New Jersey. Each video has a size range from 38GB to 58GB and is about 24 hours long. The position of this camera is very informative since it captures complex road conditions and it records videos both day and night. We specifically pick a location in which the traffic situation is difficult. This will give us more observation of “near misses” collisions and traffic violations. There are risk factors not only during busy hours, but also during night when the light conditions are bad or even during free hours when participants are more likely to violate traffic laws. 24-hour monitoring enables us to catch diverse risk factors and avoid data bias. The position also has relatively busy traffic and a diverse community. This also enhances the quality of our dataset. Python 3.8.3 will be used as the main programming language to perform object recognition and feature extraction.

    II. Object Detection
A cascade model is used to perform the object detection on the raw videos. The cascade model is composed of an existing YOLO v3 model [11] and a self-trained customized model. It is designed in such a way to retain both the pre-train YOLO labels and the customized labels (i.e., scooter rider
and biker). As seen in Figure 1, the cascade model takes a raw video as an input, then it first passes through the YOLO v3 algorithm to detect any non-person labels. If a person label is detected, then the cascade model will expand the image region where the bounding box for the person label is
detected and feed into the customized model. If the customized model detects any biker or scooter rider, it will then return the biker and scooter rider label accordingly with the detected bounding box and confidence score. If the customized model did not detect any scooter rider or biker, then the cascade model will then use the person label, bounding box, and confidence score obtained from the previous step from the YOLO v3 model. 
    ![Figure 1](/Supplement_Images/cascade model.png)

The YOLO v3 model contained the pre-train weights and neural network configuration files that was trained on the COCO dataset [12]. The YOLO v3 model is capable of detecting 80 different class labels which include these traffic relevant objects such as traffic lights, car, bicycle, person, skateboard, motorbike, bus, train, and truck. However, the YOLO v3 model lacks the scooter rider,biker labels and cannot differentiate between biker, scooter rider, and pedestrian within the person label, which are the essential part of this project. Hence, a customized model was trained to specifically detects scooter rider and biker.

The customized model was trained using the Darknet framework. To obtain the training dataset, a script was developed to crop images containing bikers or scooter riders from one raw 24-hour video. To supplement the cropped video images for sufficient training sample, online images searched through the Google engine was also used as part of the training sample. A total of 1,150 training images were labelled using the open source LabelImg application [13]. To make sure the
customized model can still perform object detection despite various video resolutions, a separate script was written to generate 10 different resolutions of the same image with 10% incrementalimage resolution, as shown i. Altogether, 10,350 images were feed into training the
customized model.

## Results:

