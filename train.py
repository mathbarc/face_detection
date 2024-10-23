import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
import models



def net_sample_output():

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


def test_loss():
    mean = 0
    count = 0
    testCriterion = nn.MSELoss()
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
        
        loss = testCriterion(output_pts, key_pts)
        mean += loss.item()
        count += 1
        
    return mean / count


def train_net(n_epochs):

    # prepare the net for training
    net.train(True)
    training_loss = []
    testing_loss = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        counter=0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
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
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            counter += 1
        tLoss = test_loss()
        training_loss.append(running_loss/counter)
        testing_loss.append(tLoss)
        print('Epoch: {}, Batch: {}, Avg. Loss: {}, Avg. Testing Loss: {}'.format(epoch + 1, batch_i+1, running_loss/counter, tLoss))
        running_loss = 0.0
        counter=0
    net.train(False)
    print('Finished Training')
    return training_loss, testing_loss


if __name__ == "__main__":
    batch_size = 32
    net = models.FaceDetectionNet()
    print(net)

    if(torch.cuda.is_available()):
        net = net.cuda()

    ## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
    # order matters! i.e. rescaling should come before a smaller crop
    data_transform = transforms.Compose( [Rescale((200,200)), RandomCrop((100,100)), Normalize(), ToTensor()])

    # testing that you've defined a transform
    assert(data_transform is not None), 'Define a data_transform'

    # create the transformed dataset
    data_path = "/data/ssd1/Datasets/Faces/"
    transformed_dataset = FacialKeypointsDataset(csv_file=data_path+'/training_frames_keypoints.csv',
                                                root_dir=data_path+'/training/',
                                                transform=data_transform)


    print('Number of images: ', len(transformed_dataset))

    # iterate through the transformed dataset and print some stats about the first few samples
    for i in range(4):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size(), sample['image'].device, sample['keypoints'].device)
        

    train_loader = DataLoader(transformed_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    test_dataset = FacialKeypointsDataset(csv_file=data_path+'/test_frames_keypoints.csv',
                                                root_dir=data_path+'/test/',
                                                transform=data_transform)


    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)


    # test the model on a batch of test images
            
    # call the above function
    # returns: test images, test predicted keypoints, test ground truth keypoints
    test_images, test_outputs, gt_pts = net_sample_output()

    # print out the dimensions of the data to see if they make sense
    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())


    # call it
    visualize_output(test_images, test_outputs, gt_pts, 4)

    ## TODO: Define the loss and optimization
    import torch.optim as optim

    ## TODO: specify loss function
    criterion = nn.MSELoss()

    ## TODO: specify optimizer
    optimizer = optim.ASGD(net.parameters(),lr=0.1)

    # train your network
    n_epochs = 1000 # start small, and increase when you've decided on your model structure and hyperparams

    # this is a Workspaces-specific context manager to keep the connection
    # alive while training your model, not part of pytorch
    #with active_session():
    #    train_net(n_epochs)
    training_loss, testing_loss = train_net(n_epochs)

    plt.plot(range(1,len(training_loss)+1), training_loss)
    plt.plot(range(1,len(testing_loss)+1), testing_loss)
    plt.legend(["train", "test"])

    # get a sample of test data again
    test_images, test_outputs, gt_pts = net_sample_output()

    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())

    visualize_output(test_images, test_outputs, gt_pts,4)

    ## TODO: change the name to something uniqe for each new model
    model_dir = './'
    model_name = 'keypoints_model_1.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)

