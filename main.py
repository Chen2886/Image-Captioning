import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from torchvision import transforms
 
 
# Load MS COCO data, and use API for annotations and caption
coco = COCO('./cocoapi/annotations/instances_val2014.json')
coco_caps = COCO('./cocoapi/annotations/captions_val2014.json')
 
# all images ids
ids = list(coco.anns.keys())
 
# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
 
# Set the minimum word count threshold.
vocab_threshold = 5
 
# Specify the batch size.
batch_size = 10
 
# Load the data
data_loader = torch.utils.data.DataLoader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)