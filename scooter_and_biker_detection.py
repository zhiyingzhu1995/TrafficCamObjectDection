# -*- coding: utf-8 -*-
"""Scooter and Biker detection.ipynb

Automatically generated by Colaboratory.


"""


"""**Connect google drive**"""

# Check if NVIDIA GPU is enabled
!nvidia-smi

from google.colab import drive
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive

"""**1) Clone, configure & compile Darknet**"""

# Clone

!git clone https://github.com/AlexeyAB/darknet

# Commented out IPython magic to ensure Python compatibility.

!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile



"""# New Section"""

# Compile
!make



"""**2) Configure yolov3.cfg file**"""

# Make a copy of yolov3.cfg
!cp cfg/yolov3.cfg cfg/yolov3_training.cfg

# Change lines in yolov3.cfg file for 2 class
!sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
!sed -i 's/max_batches = 500200/max_batches = 4000/' cfg/yolov3_training.cfg # class * 2000, 82 classes *2000 = 164000
!sed -i '610 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '696 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '783 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '603 s@filters=255@filters=21@' cfg/yolov3_training.cfg # filters = (classes+5)*3 = (82+5)*3 = 261
!sed -i '689 s@filters=255@filters=21@' cfg/yolov3_training.cfg
!sed -i '776 s@filters=255@filters=21@' cfg/yolov3_training.cfg



"""**3) Create .names and .data files**"""

!echo -e 'Biker\nScooter Rider' > data/obj.names

!echo -e 'classes= 2\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = /mydrive/yolov3' > data/obj.data


"""**4) Save yolov3_training.cfg and obj.names files in Google drive**"""

!cp cfg/yolov3_training.cfg /mydrive/yolov3/yolov3_testing.cfg
!cp data/obj.names /mydrive/yolov3/classes.txt

"""**5) Create a folder and unzip image dataset**"""

!mkdir data/obj
!unzip /mydrive/yolov3/images.zip -d data/obj


"""**6) Create train.txt file**"""

import glob

images_list = glob.glob("/content/darknet/data/obj/images/*.jpeg")

print(len(images_list))
with open("data/train.txt", "w") as f:
    f.write("\n".join(images_list))

"""**7) Download pre-trained weights for the convolutional layers file**"""

!wget https://pjreddie.com/media/files/darknet53.conv.74

"""**8) Start training**"""

!./darknet detector train data/obj.data cfg/yolov3_training.cfg darknet53.conv.74 -dont_show
# Uncomment below and comment above to re-start your training from last saved weights
# !./darknet detector train data/obj.data cfg/yolov3_training.cfg /mydrive/yolov3/yolov3_training_last.weights -dont_show
