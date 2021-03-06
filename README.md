# TrafficCamObjectDection

[Final Report](https://github.com/zhiyingzhu1995/TrafficCamObjectDection/blob/c3c067bd3e3723ad827b6141282204b96abf5db5/Report/Capstone%20Final%20Project%20Report_Zhiying_Zhu.pdf)

[YOLOV3 Weights](https://ccnymailcuny-my.sharepoint.com/:u:/g/personal/zzhu000_citymail_cuny_edu/EcffKRVygq9Eh1uiqdAJKKwBjFYKlgj07mYFqJCNKBYQVg?e=rAylDT) 

[YOLOV3 CFG](https://ccnymailcuny-my.sharepoint.com/:u:/g/personal/zzhu000_citymail_cuny_edu/ER1GUNNitnVNkzQotG55n7ABTxaa4Nysf1rMcpKOUBrj6w?e=zaygl2)

[YOLOV3 Label](https://ccnymailcuny-my.sharepoint.com/:t:/g/personal/zzhu000_citymail_cuny_edu/EfIGqKVAkexEiJdmeJnLwscBL4HBbEkeztW1oRLUUM3cHA?e=WU5kRe) 

[Customized Model Weights](https://ccnymailcuny-my.sharepoint.com/:u:/g/personal/zzhu000_citymail_cuny_edu/Eb-FPcjjHFtDqnjh12ovXGkBUijQbx_OuUe_o_WGlqd39Q?e=PSKNJL) 

[Customized Model CFG](https://ccnymailcuny-my.sharepoint.com/:u:/g/personal/zzhu000_citymail_cuny_edu/EaZSXapdL_tAjZh2wQG9A3QBLR9TPBocGALzWEKTt2s6qA?e=ffATr9) 

[Customized Model Label](https://ccnymailcuny-my.sharepoint.com/:t:/g/personal/zzhu000_citymail_cuny_edu/EcZeqIcZclBGrVxXw19qmiQBYvBlfes1DOIyybkdpXyOiA?e=Ks6Qft) 

[Training Image Files with Label Txt File](https://ccnymailcuny-my.sharepoint.com/:u:/g/personal/zzhu000_citymail_cuny_edu/ESzyHiT3pDpIqp_MqmQsXpABpKwBzPncFAtjJhkT2qIl7g?e=bRtHc2)

[Testing Biker Image Files with Label Txt File](https://ccnymailcuny-my.sharepoint.com/:f:/g/personal/zzhu000_citymail_cuny_edu/EkENDBiTnihKhk00e1VZq9YBk_dlmmt_fbQT5YE0zgAvSQ?e=TKUDGO)

[Testing Scooter Image Files with Label Txt File](https://ccnymailcuny-my.sharepoint.com/:f:/g/personal/zzhu000_citymail_cuny_edu/EmKcI4zRzLJAs_oMWW2v74YBlOLgKAdbmzLGMCndX9wzTA?e=15dKK2)

*Password for these files: cam*

## Design:
    I. Data Collection
To understand the real traffic situation, a camera is replaced to record the 24-hour traffic interactions at Asbury Ave, New Jersey. Each video has a size range from 38GB to 58GB and is about 24 hours long. The position of this camera is very informative since it captures complex road conditions and it records videos both day and night. We specifically pick a location in which the traffic situation is difficult. This will give us more observation of ???near misses??? collisions and traffic violations. There are risk factors not only during busy hours, but also during night when the light conditions are bad or even during free hours when participants are more likely to violate traffic laws. 24-hour monitoring enables us to catch diverse risk factors and avoid data bias. The position also has relatively busy traffic and a diverse community. This also enhances the quality of our dataset. Python 3.8.3 will be used as the main programming language to perform object recognition and feature extraction.

    II. Object Detection
A cascade model is used to perform the object detection on the raw videos. The cascade model is composed of an existing YOLO v3 model [11] and a self-trained customized model. It is designed in such a way to retain both the pre-train YOLO labels and the customized labels (i.e., scooter rider
and biker). As seen in [Figure 1], the cascade model takes a raw video as an input, then it first passes through the YOLO v3 algorithm to detect any non-person labels. If a person label is detected, then the cascade model will expand the image region where the bounding box for the person label is
detected and feed into the customized model. If the customized model detects any biker or scooter rider, it will then return the biker and scooter rider label accordingly with the detected bounding box and confidence score. If the customized model did not detect any scooter rider or biker, then the cascade model will then use the person label, bounding box, and confidence score obtained from the previous step from the YOLO v3 model. 
    
**Figure 1**: ![Figure 1](https://github.com/zhiyingzhu1995/TrafficCamObjectDection/blob/fd96c376035984cf41e6658eb72efef60d8e5e82/Supplement_Images/cascade%20model.png?raw=true "Figure 1") 

The YOLO v3 model contained the pre-train weights and neural network configuration files that was trained on the COCO dataset [12]. The YOLO v3 model is capable of detecting 80 different class labels which include these traffic relevant objects such as traffic lights, car, bicycle, person, skateboard, motorbike, bus, train, and truck. However, the YOLO v3 model lacks the scooter rider,biker labels and cannot differentiate between biker, scooter rider, and pedestrian within the person label, which are the essential part of this project. Hence, a customized model was trained to specifically detects scooter rider and biker.

The customized model was trained using the Darknet framework. To obtain the training dataset, a script was developed to crop images containing bikers or scooter riders from one raw 24-hour video. To supplement the cropped video images for sufficient training sample, online images searched through the Google engine was also used as part of the training sample. A total of 1,150 training images were labelled using the open source LabelImg application [13]. To make sure the
customized model can still perform object detection despite various video resolutions, a separate script was written to generate 10 different resolutions of the same image with 10% incrementalimage resolution, as shown in [Figure 2]. Altogether, 10,350 images were feed into training the
customized model.

**Figure 2**: ![Figure 2](https://github.com/zhiyingzhu1995/TrafficCamObjectDection/blob/0233d5d75b378daac174354171117ff9a4a150d6/Supplement_Images/10%20diff%20resolution.png) 

## Results:
As shown in [Figure 3], you can see that the cascade model can successfully detect the desirable labels of biker and scooter riders from our videos. It can compute the correct class label and draw appropriate bounding boxes on detected object on top of the image. From [Figure 4], you can see that the cascade model is a decent method for retaining the pretrain YOLO v3 models (i.e., traffic light, car, motorbike) and obtaining the new labels (i.e., biker and scooter rider) from the customized model as well as overlaying all label and bounding boxes on the same image.

The model performance and evaluation of the customized model are showed in [Table 1]. The algorithm is evaluated against a total of 374 biker and scooter rider images combined. Out of the 374 images, there are with 218 biker images and 256 scooter rider images. The ratio of images to the number of bounding boxes is not 1:1 because there are images with multiple objects detected. 218 biker images have 272 bounding boxes; 156 scooter rider images have 211 bounding boxes; and 374 biker and scooter rider images combined have 483 bounding boxes. As you see in [Table 1], the customized model has 94.85% accuracy in detecting biker label, 83.89% accuracy in detecting scooter rider label, and 90.06% accuracy in detecting either biker or scooter rider label. The model has a 15% chance of missing a scooter rider label, 5% chance of missing a biker label, and 9% chance of missing either a biker or a scooter label. The average intersection over union (IoU) is 95% for the biker label, 93.7% for scooter rider labels, and 94.5% for either biker or scooter label. 

**Figure 3**: ![Figure 3](https://github.com/zhiyingzhu1995/TrafficCamObjectDection/blob/0233d5d75b378daac174354171117ff9a4a150d6/Supplement_Images/result1.png) 

**Figure 4**: ![Figure 4](https://github.com/zhiyingzhu1995/TrafficCamObjectDection/blob/0233d5d75b378daac174354171117ff9a4a150d6/Supplement_Images/result%202.png) 

**Table 1**: 


<img src="https://github.com/zhiyingzhu1995/TrafficCamObjectDection/blob/0233d5d75b378daac174354171117ff9a4a150d6/Supplement_Images/Final%20Eval%20Results.png" width = "650" height = "400">


## References
[1] 	N. H. T. S. Administration, "Traffic safety facts: 2018 data: Pedestrians," Annals of Emergency Medicine, 2018.

[2] 	N. H. T. S. Administration, "Traffic safety facts: 2015 data: Pedestrians," Annals of Emergency Medicine, 2015.

[3] 	N. H. T. S. Administration, "Traffic safety facts: 2014 data: Pedestrians," Annals of Emergency Medicine, 2014.

[4] 	K. D., "Pedestrian and bicycle volume data collection using drone technology," J Urban Technol, vol. 27(2), pp. 45-60, 2020. 

[5] 	P. P. Hatfield J, "An investigation of behaviour and attitudes relevant to the user safety of pedestrian/cyclist shared paths," Transp Res Part F, pp. 35-47, 2016. 

[6] 	B. A. G. G. Gkekas F, "Perceived safety and experienced incidents between pedestrians and cyclists in a high-volume non-motorized shared space," Transp Res Interdiscip Perspect, 2020. 

[7] 	D. a. K. P. Kim, "Analysis of potential collisions between pedestrians and personal transportation devices in a university campus: an application of unmanned aerial vehicles," Journal of American college health, 2021. 

[8] 	J. &. F. A. Redmon, "Yolov3: An incremental improvement," arXiv preprint arXiv, p. 804.02767. 

[9] 	"OpenCV Library," [Online]. Available: https://opencv.org/. [Accessed 23 09 2021].

[10] 	L. W. e. al, "SSD: Single Shot MultiBox Detector," European conference on computer vision, vol. 1512.02325, pp. pp. 21-37, 2016, October. 

[11] 	P. a. J. M. Viola, "Robust real-time face detection," International journal of computer vision, vol. 57(2), pp. 37-154, 2004. 

[12] 	L. Dinalankara, "Face detection & face recognition using open computer vision classifies," ResearchGate, 2017. 

[13] 	G. &. H. T. Levi, "Age and gender classification using convolutional neural networks," Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 34-42, 2015. 

