import os,glob
import numpy as np
from PIL import Image
import shutil


def main():
    
    ##########
    # generate different image resolution 
    # and create correponding txt file with rename image
    ###########
    image_path = <'fill in your image location'>
    os.chdir(image_path)

    # get the name of the file in that folder
    fileList = []
    for file in glob.glob("*.jpeg"):
        fileList.append(file)

    # iterate each image in that folder and generate 10 different resolution 
    for file in fileList:
        image_path = file
        filename = image_path.split(".")[0]
        try:
            image_file = Image.open(image_path)
            for number in range(10,100,10):
                image_file.save(filename+"_quality_"+str(number)+".jpeg", quality=number)
        except:
            pass
    print("finish generating different resolution of images")

    #############
    #create responding txt file with renmae
    #for each manually create image resolution
    #############
    
    # get a list of txt file and keep adding new txt files for the images with change of resolution
    for _,file in enumerate(os.listdir(image_path)):
        filename = file.split(".")[0]
        if ".txt" in file:
            for number in range(10,100,10):
                try:
                    shutil.copyfile(path+'/'+file, path+'/'+filename + "_quality_"+str(number)+".txt")
                except:
                    pass
    print("finish generating txt files for different resolution of images")

if __name__ == "__main__":
    main()
    
            
