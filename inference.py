from math import factorial
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import torch
import torch.nn as nn
import torch.nn.functional as F
from models import FaceDetectionNet
from utils import show_all_keypoints
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
    


def detectFace(image, net, face_cascade):

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    # make a copy of the original image to plot detections on
    image_copy = image.copy()

    results = []
    # loop over the detected faces from your haar cascade
    for rect in faces:
        (x,y,w,h) = rect
        
        scale = 1.

        w_diff = abs(scale-1)*w
        h_diff = abs(scale-1)*h
        
        w = int(min(scale * w, image.shape[1]))
        h = int(min(scale * h, image.shape[0]))

        x = int(max(x - w_diff/2., 0))
        y = int(max(y - h_diff/2., 0))


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
        #print(output.shape)
        if(torch.cuda.is_initialized()):
            output = output.cpu()

        #plt.figure(figsize=(20,10))
        #ax = plt.subplot(1, 2, i+1)
        #i+=1
        ## TODO: Display each detected face and the corresponding keypoints        
        out = output.detach().numpy()
        out = (out+1.)*.5
        #out_draw = out.copy()

        #out_draw = out_draw * [roi.shape[1], roi.shape[0]]
        #show_all_keypoints(roi,out_draw)
        #cv2.imshow("roi", roi)
        #cv2.waitKey()

        out[:,0] = (out[:,0]*w)+x
        out[:,1] = (out[:,1]*h)+y
        results.append((numpy.array([x,y,w,h],numpy.int32), out))
        
    return results


if __name__ == "__main__":


    # load in color image for face detection
    cap = cv2.VideoCapture(0)
    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

    #net = FaceDetectionNet()

    ## TODO: load the best saved model parameters (by your path name)
    ## You'll need to un-comment the line below and add the correct name for *your* saved model
    # net.load_state_dict(torch.load('keypoints_model_1.pt'))
    net = torch.load('keypoints_model_1.pt')
    net = net.double()

    ## print out your net and prepare it for testing (uncomment the line below)
    net.eval()

    if torch.cuda.is_available():
        print("using cuda:", torch.cuda.get_device_name())
        torch.cuda.init()
    if(torch.cuda.is_initialized()):
        net = net.cuda()
        
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    key = 0
    while key != 27:
        ret, image = cap.read()
        if not ret:
            break
        #image = image[:,900:]
        results = detectFace(image, net, face_cascade)

        for (rect, face_points) in results:
            show_all_keypoints(image, face_points)
            cv2.rectangle(image, rect[:2], rect[:2]+rect[2:4], (0,255,0),3)

        cv2.imshow("result", image)
        key = cv2.waitKey(1)
            
        
