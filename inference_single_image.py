
import torch
from utils import show_all_keypoints
import cv2
import numpy
import time
from inference import toTensor
from models import FaceDetectionNetV2

def inference_image(net, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        resized = cv2.resize(gray, (100,100)).astype(numpy.double)
        resized = resized/255.0

        #print(resized.shape)
        
        ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        netInput = toTensor(resized)
        
        ## TODO: Make facial keypoint predictions using your loaded, trained network   
        #print(netInput.shape)
        start_time = time.time()
        output = net(netInput)
        end_time = time.time()

        print(end_time-start_time, "s")

        output = output.view(68, 2)
        #print(output.shape)
        if(torch.cuda.is_initialized()):
            output = output.cpu()

        #plt.figure(figsize=(20,10))
        #ax = plt.subplot(1, 2, i+1)
        #i+=1
        ## TODO: Display each detected face and the corresponding keypoints        
        out = output.detach().numpy()

        print(out[:,0].min(), out[:,0].max())
        print(out[:,1].min(), out[:,1].max())
        print()
    
        out = (out * 0.5) + 0.5
        print(out[:,0].min(), out[:,0].max())
        print(out[:,1].min(), out[:,1].max())
        print()
        
        out *= [image.shape[1], image.shape[0]]
        print(out[:,0].min(), out[:,0].max())
        print(out[:,1].min(), out[:,1].max())
        print()
        return out


if __name__ == "__main__":


    # load in color image for face detection
    img = cv2.imread("/data/ssd1/Datasets/Faces/test/Ben_Stein_10.jpg")
    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

    #net = FaceDetectionNet()

    ## TODO: load the best saved model parameters (by your path name)
    ## You'll need to un-comment the line below and add the correct name for *your* saved model
    # net.load_state_dict(torch.load('keypoints_model_1.pt'))
    net = FaceDetectionNetV2()
    net.load_state_dict(torch.load('keypoints_model_1.pt'))
    print(net)
    net = net.double()

    ## print out your net and prepare it for testing (uncomment the line below)
    net.eval()

    if torch.cuda.is_available():
        print("using cuda:", torch.cuda.get_device_name())
        torch.cuda.init()
    if(torch.cuda.is_initialized()):
        net = net.cuda()
     
    key_pts = inference_image(net,img)

    show_all_keypoints(img, key_pts)

    cv2.imshow("result", img)
    key = cv2.waitKey(0)
            
        
