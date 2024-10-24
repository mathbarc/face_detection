import glob
import os
from matplotlib.pyplot import sca
from scipy.sparse import data
import torch
from torch.utils.data import Dataset, DataLoader
import numpy
import matplotlib.image as mpimg
import pandas as pd
import cv2
from utils import show_all_keypoints
import random

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = cv2.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = numpy.copy(image)
        key_pts_copy = numpy.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy-50)/50.

        return {'image': image_copy, 'keypoints': key_pts_copy}

class CropFace(object):

    def __init__(self, scale=1.) -> None:
        if isinstance(scale, tuple) or isinstance(scale, list):
            self.scale_min = scale[0]
            self.scale_max = scale[1]
        else:
            self.scale = scale


    def __call__(self, sample):

        image, key_pts = sample["image"], sample["keypoints"]

        face_rect = cv2.boundingRect(key_pts.astype(numpy.int32))

        x,y,w,h = face_rect

        if hasattr(self,"scale_min") and hasattr(self,"scale_max"):
            scale = (random.randrange(0,100,10) * (self.scale_max - self.scale_min) / 100) + self.scale_min
        else:
            scale = self.scale

        w_diff = abs(1.-scale)*w
        h_diff = abs(1.-scale)*h
        
        w = int(min(scale * w, image.shape[1]))
        h = int(min(scale * h, image.shape[0]))
        x = int(max(x - w_diff/2., 0))
        y = int(max(y - h_diff/2., 0))


        face = image[y:y+h, x:x+w]
        key_pts = key_pts + [-x,-y]

        return {"image":face, "keypoints":key_pts}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}
    

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        #print(h, new_h)
        #print(w, new_w)

        top = numpy.random.randint(0, h - new_h)
        left = numpy.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        image_dataset = torch.from_numpy(image)
        key_pts_dataset = torch.from_numpy(key_pts)
        
        return {'image': image_dataset,
                'keypoints': key_pts_dataset}


if __name__ == "__main__":

    from torchvision.transforms import Compose
    dataset = FacialKeypointsDataset("/data/ssd1/Datasets/Faces/training_frames_keypoints.csv", "/data/ssd1/Datasets/Faces/training", Compose([CropFace((1.,5.)), Rescale((100,100)), Normalize()]))

    

    for sample in dataset:
        img, key_pts = sample["image"], sample["keypoints"]
        
        key_pts = key_pts * [50,50]
        key_pts = key_pts + [50,50]


        show_all_keypoints(img, key_pts)
        cv2.imshow("face", img)
        key = cv2.waitKey()
        if key == 27:
            break
    



