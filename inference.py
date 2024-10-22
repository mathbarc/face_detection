import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import torch
import torch.nn as nn
import torch.nn.functional as F
from models import FaceDetectionNet

import cv2


def toTensor(img):
    image = numpy.copy(img)
    if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(1,image.shape[0], image.shape[1], 1)
            
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((0, 3, 1, 2))
    image_tensor = torch.from_numpy(image)

    if(torch.cuda.is_initialized()):
        image_tensor = image_tensor.cuda()
    return image_tensor
    
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    imgSize = image.shape

    if(len(imgSize) == 3):
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(show)
    else:
        show = image
        plt.imshow(show, cmap="gray")
    
    
    plt.scatter((predicted_key_pts[:, 0]), (predicted_key_pts[:, 1]), s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


if __name__ == "__main__":


    # load in color image for face detection
    image = cv2.imread('data/ssd1/Datasets/Faces/training/Abdullah_Gul_10.jpg')
    showImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plot the image
    cv2.imshow(showImg)
    cv2.waitKey()

    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    # make a copy of the original image to plot detections on
    image_with_detections = showImg.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

    cv2.imshow(image_with_detections)
    cv2.waitKey()


    net = FaceDetectionNet()

    ## TODO: load the best saved model parameters (by your path name)
    ## You'll need to un-comment the line below and add the correct name for *your* saved model
    net.load_state_dict(torch.load('keypoints_model_1.pt'))
    net = net.double()

    ## print out your net and prepare it for testing (uncomment the line below)
    net.eval()

    if torch.cuda.is_available():
        print("using cuda:", torch.cuda.get_device_name())
        torch.cuda.init()
    if(torch.cuda.is_initialized()):
        net = net.cuda()
        
        

            
            
    image_copy = image.copy()

    # loop over the detected faces from your haar cascade
    i=0
    for (x,y,w,h) in faces:
        
        #print(x,y)
        # Select the region of interest that is the face in the image 
        roi = image_copy[y:y+h, x:x+w]
        
        ## TODO: Convert the face region from RGB to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        resized = cv2.resize(gray, (100,100)).astype(float)
        resized = resized/255.0

        #print(resized.shape)
        
        ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        netInput = toTensor(resized)
        
        ## TODO: Make facial keypoint predictions using your loaded, trained network   
        #print(netInput.shape)
        output = net(netInput)
        output = output.view(68, 2)
        output = output
        #print(output.shape)
        if(torch.cuda.is_initialized()):
            output = output.cpu()

        #plt.figure(figsize=(20,10))
        #ax = plt.subplot(1, 2, i+1)
        #i+=1
        ## TODO: Display each detected face and the corresponding keypoints        
        out = output.detach().numpy()
        out = out*100.+50.
        out[:,0] = (out[:,0]*(w/100.))+x
        out[:,1] = (out[:,1]*(h/100.))+y
        
        show_all_keypoints(image, out)

    plt.show()
        