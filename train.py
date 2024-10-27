from random import Random
from cv2 import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor, CropFace, CropFaceHaar

import mlflow
import mlflow.pytorch

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
import models
import sys
from schedulers import RampUpCosineDecayScheduler
import logging
import tqdm

def net_sample_output(test_loader):

    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        if(torch.cuda.is_initialized()):
            images = images.type(torch.cuda.FloatTensor)
        else:
            images = images.type(torch.FloatTensor)        

        # forward pass to get net output
        output_pts = net(images)
        

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after first image is tested
        if i == 0:
            return images.cpu(), output_pts, key_pts
        
        


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data.cpu()   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu().numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts*100.+50.

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts*100.+50.

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
        #print(predicted_key_pts[0])

    plt.show()


def loss_function(pred:torch.Tensor, target:torch.Tensor):
    errors = 1.-F.cosine_similarity(pred, target)
    return torch.mean(errors, dim=0)


def test_loss(test_loader):
    mean = 0
    count = 0
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']
        
        key_pts = key_pts.view(key_pts.size(0), -1)
        if(torch.cuda.is_initialized()):
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)
        else:
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

        # convert images to FloatTensors
        if(torch.cuda.is_initialized()):
            images = images.type(torch.cuda.FloatTensor)
        else:
            images = images.type(torch.FloatTensor)        

        # forward pass to get net output
        output_pts = net(images)
        
        loss = loss_function(output_pts, key_pts)
        mean += loss.item()
        count += 1
        
    return mean / count


def train_net(n_epochs, batch_size, lr):


    train_loader = DataLoader(transformed_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)

    # prepare the net for training
    net.train(True)
    training_loss = []
    testing_loss = []

    ## TODO: specify optimizer
    optimizer = optim.Adam(net.parameters(),lr=lr)

    scheduler = RampUpCosineDecayScheduler(optimizer, lr, 1e-5, 1000, 10)

    training_params = {
        "lr" : lr,
        "batch_size" : batch_size,
        "n_epochs" : n_epochs
    }
    
    mlflow.set_tracking_uri("http://mlflow.cluster.local")
    experiment = mlflow.get_experiment_by_name("Face Detection")
    if experiment is None:
        experiment_id = mlflow.create_experiment("Face Detection")
    else:
        experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.log_params(training_params)

    best_test_loss = sys.float_info.max
    for epoch in tqdm.tqdm(range(n_epochs)):  # loop over the dataset multiple times

        running_loss = 0.0
        counter=0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            if(torch.cuda.is_available()):
                key_pts = key_pts.type(torch.cuda.FloatTensor)
                images = images.type(torch.cuda.FloatTensor)
            else:
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = loss_function(output_pts, key_pts)


            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            counter += 1
        tLoss = test_loss(test_loader)
        training_loss.append(running_loss/counter)
        testing_loss.append(tLoss)
        metrics = {
            "training_loss":running_loss/counter,
            "testing_loss":tLoss,
            "lr":scheduler.get_last_lr()
        }
        scheduler.step()
        mlflow.log_metrics(metrics,epoch+1)
        #print('Epoch: {}, Batch: {}, Avg. Loss: {}, Avg. Testing Loss: {}'.format(epoch + 1, batch_i+1, running_loss/counter, tLoss))
        mlflow.pytorch.log_model(net, "last",extra_files=["./models.py"])
        if tLoss < best_test_loss:
            best_test_loss = tLoss
            mlflow.pytorch.log_model(net, "best",extra_files=["./models.py"])
        running_loss = 0.0
        counter=0

    net.train(False)
    print('Finished Training')
    return training_loss, testing_loss


if __name__ == "__main__":
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    batch_size = 32

    net = models.FaceDetectionNetV2()
    print(net)

    if(torch.cuda.is_available()):
        net = net.cuda()

    ## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
    # order matters! i.e. rescaling should come before a smaller crop
    data_transform = transforms.Compose( [Rescale((130,130)), RandomCrop((100,100)), Normalize(), ToTensor()])

    # testing that you've defined a transform
    assert(data_transform is not None), 'Define a data_transform'

    # create the transformed dataset
    data_path = "/data/ssd1/Datasets/Faces/"
    transformed_dataset = FacialKeypointsDataset(csv_file=data_path+'/training_frames_keypoints.csv',
                                                root_dir=data_path+'/training/',
                                                transform=data_transform)


    print('Number of images: ', len(transformed_dataset))

        
 
    test_transform = transforms.Compose( [Rescale((130,130)), RandomCrop((100,100)), Normalize(), ToTensor()])
    test_dataset = FacialKeypointsDataset(csv_file=data_path+'/test_frames_keypoints.csv',
                                                root_dir=data_path+'/test/',
                                                transform=test_transform)

    # train your network
    n_epochs = 1000 # start small, and increase when you've decided on your model structure and hyperparams

    # this is a Workspaces-specific context manager to keep the connection
    # alive while training your model, not part of pytorch
    #with active_session():
    #    train_net(n_epochs)
    training_loss, testing_loss = train_net(n_epochs, batch_size, 0.001)

    ## TODO: change the name to something uniqe for each new model
    model_dir = './'
    model_name = 'keypoints_model_1.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)

