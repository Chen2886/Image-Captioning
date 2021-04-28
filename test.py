import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
import time
import json
import math

from pycocotools.coco import COCO

import nltk

import skimage.io as io
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

from vocab import Vocab
from model import EncoderCNN, DecoderRNN
from utils import train, validate, save_epoch, early_stopping, clean_sentence, get_prediction


###############################################################################
###############################################################################
# Testing
###############################################################################
###############################################################################

test_file_path = './cocoapi/images/test2014'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_captions(image_id, coco_caps):
    annIds = coco_caps.getAnnIds(imgIds=image_id)
    anns = coco_caps.loadAnns(annIds)
    return anns

class transform_dataset(data.Dataset):

    def __init__(self, coco, annotations_file, image_folder, batch_size, mode):
        self.vocab = Vocab(annotations_file)
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.mode = mode
        if not self.mode == 'test':
            self.coco_caps = COCO(os.path.join(annotations_file))
            self.coco = coco
            self.index = list(self.coco.anns.keys())

            '''
            Get the length of all the captions available in the set. Needed for training
            the model and to get only images with captions of a fixed length
            '''
            all_tokens = []
            for index in tqdm(np.arange(len(self.index))):
                test_id = self.coco.anns[self.index[index]]['image_id']
                captions = load_captions(test_id, self.coco_caps)[0]['caption']
                all_tokens.append(nltk.tokenize.word_tokenize(str(captions).lower()))

            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]

    def __len__(self):
        if not self.mode == 'test':
            return len(self.index)
        else:
            return len(self.paths)

    def __getitem__(self, id):
        if not self.mode == 'test':
            # get the id of the image located in self.index and preprocess it
            img_id = self.coco.anns[self.index[id]]['image_id']
            img = self.coco.loadImgs(img_id)[0]
            image = Image.open(os.path.join(self.image_folder, img['file_name'])).convert('RGB')
            image = self.transform(image)

            # get the captions associated with the image id
            captions = load_captions(img_id, self.coco_caps)[0]['caption']
            caption_arr = []
        
            # tokenize and vectorize the captions
            tokens = nltk.tokenize.word_tokenize(str(captions).lower())
            caption_arr.append(self.vocab(self.vocab.start_word))
            caption_arr.extend([self.vocab(token) for token in tokens])
            caption_arr.append(self.vocab(self.vocab.end_word))
                
            caption_arr = torch.Tensor(caption_arr).long()

            return image, caption_arr
        else:
            path = self.paths[id]
            # print(path)
            image = Image.open(os.path.join(self.image_folder, path)).convert('RGB')
            # orig = np.array(image)
            image = self.transform(image)
            # print(image.size())
            return image

    def get_indices(self):
        '''
        This function returns a subset of the dataset of size batch_size where the length of the captions are equal to each other
        This is necessary because the caption tensors need to be of the same length
        '''
        sel_length = np.random.choice(self.caption_lengths)
        all_indicies = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indicies = list(np.random.choice(all_indicies, size=self.batch_size))
        return indicies
    

    # Define a transform to pre-process the training images.
    transform = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.CenterCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))])

# hyperparameters
BATCH_SIZE = 32
VOCAB_THRESHOLD = 5
EPOCHS = 10
EMBED_SIZE = 256
HIDDEN_SIZE = 512

test_dataset = transform_dataset(coco=None, annotations_file=os.path.join('./cocoapi/annotations', 'image_info_test2014.json'), image_folder=test_file_path, batch_size=BATCH_SIZE, mode='test')

 
# Load the data
test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=0)


# transform = transforms.Compose([ 
#         transforms.Resize(256),                          # smaller edge of image resized to 256
#         transforms.CenterCrop(224),                      # get 224x224 crop from random location
#         transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
#         transforms.ToTensor(),                           # convert the PIL Image to a tensor
#         transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                             (0.229, 0.224, 0.225))])

# image = Image.open(os.path.join(test_file_path, 'COCO_test2014_000000097114.jpg')).convert('RGB')
# image = transform(image)
image = next(iter(test_loader))
# image = np.squeeze(image)
# image = torch.transpose(image, 1, 3)
print(image.size())
# print('here')
# transformed_image = image.numpy()
# transformed_image = np.squeeze(transformed_image)
# transformed_image = transformed_image.transpose((1, 2, 0))

checkpoint = torch.load(os.path.join('./models', 'train-model-018900.pkl'))
vocab = test_loader.dataset.vocab
vocab_size = len(vocab)

encoder = EncoderCNN(EMBED_SIZE)
encoder.eval()
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size)
decoder.eval()

encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    image = image.cuda()

# features = encoder(image).unsqueeze(1)
# output = decoder.sample(features)
# sentence = clean_sentence(output, vocab)
# print('example sentence:', sentence)

get_prediction(test_loader, encoder, decoder, vocab)