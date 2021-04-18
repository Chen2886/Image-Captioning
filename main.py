import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
 
 
# Load MS COCO data, and use API for annotations and caption
train_coco = COCO('./cocoapi/annotations/instances_train2014.json')
train_coco_caps = COCO('./cocoapi/annotations/captions_train2014.json')
 
# # all images ids
ids = list(train_coco.anns.keys())

train_file_path = './cocoapi/images/train2014'
val_file_path = './cocoapi/images/val2014'
test_file_path = './cocoapi/images/test2014'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define a transform to pre-process the training images.
# transform_train = transforms.Compose([ 
#     transforms.Resize(256),                          # smaller edge of image resized to 256
#     transforms.RandomCrop(224),                      # get 224x224 crop from random location
#     transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                        #  (0.229, 0.224, 0.225))])

def load_captions(image_id):
    annIds = train_coco_caps.getAnnIds(imgIds=image_id)
    anns = train_coco_caps.loadAnns(annIds)
    return train_coco_caps.showAnns(anns)
    

class transform_dataset(data.Dataset):
    # def __init__(self, images):
    #     self.img = images
    def __init__(self, indices):
        self.index = indices

    def __len__(self):
        return len(self.index)

    def __getitem__(self, id):
        # image = coco.loadImgs(self.index[id])
        img_id = train_coco.anns[self.index[id]]['image_id']
        img = train_coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(train_file_path, img['file_name'])).convert('RGB')
        image = self.transform(image)
        # test = self.transform(io.imread(os.path.join(train_file_path, img['file_name'])))
        print(load_captions(img_id))
        return np.transpose(image)

    # Define a transform to pre-process the training images.
    transform = transforms.Compose([ 
        # transforms.ToPILImage(),
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))])

dataset = transform_dataset(indices=ids) 
# Set the minimum word count threshold.
vocab_threshold = 5
 
# Specify the batch size.
batch_size = 10
 
# Load the data
data_loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=batch_size,
                                        shuffle=True)

# visualize images
count = 0
for images in data_loader:
    for image in images:
        if count < 10:
            plt.axis('off')
            plt.imshow(image)
            plt.show()
        count += 1

# data_loader = torch.utils.data.DataLoader(dataset=transform_train,
#                          mode='train',
#                          batch_size=batch_size,
#                          vocab_threshold=vocab_threshold,
#                          vocab_from_file=False)