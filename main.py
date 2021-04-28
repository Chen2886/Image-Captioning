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
# PREPROCESSING
###############################################################################
###############################################################################

# Load MS COCO data, and use API for annotations and caption
train_coco = COCO('./cocoapi/annotations/instances_train2014.json')
val_coco = COCO('./cocoapi/annotations/instances_val2014.json')

# file path to the actual images
train_file_path = './cocoapi/images/train2014'
val_file_path = './cocoapi/images/val2014'
test_file_path = './cocoapi/images/test2014'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

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
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig = np.array(image)
            image = self.transform(image)
            return orig, image

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
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
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

train_dataset = transform_dataset(coco=train_coco, annotations_file=os.path.join('./cocoapi/annotations','captions_train2014.json'), image_folder=train_file_path, batch_size=BATCH_SIZE, mode='train') 
val_dataset = transform_dataset(coco=val_coco, annotations_file=os.path.join('./cocoapi/annotations', 'captions_val2014.json'), image_folder=val_file_path, batch_size=BATCH_SIZE, mode='val')
test_dataset = transform_dataset(coco=None, annotations_file=os.path.join('./cocoapi/annotations', 'image_info_test2014.json'), image_folder=test_file_path, batch_size=BATCH_SIZE, mode='test')

 
# Load the data
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=0)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=0)

# visualize images
# count = 0
# for images in data_loader:
#     for image in images:
#         if count < 10:
#             plt.axis('off')
#             plt.imshow(image)
#             # print(caption)
#             plt.show()
#         count += 1

# get the most frequent number of caption lengths and sort in reverse order (most frequent length is index 0)
counter = Counter(train_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)

indices = train_loader.dataset.get_indices()
print('{} sampled indices: {}'.format(len(indices), indices))
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
train_loader.batch_sampler.sampler = new_sampler
vocab_size = len(train_loader.dataset.vocab)

# for batch in train_loader:
#     images, captions = batch
#     break

###############################################################################
###############################################################################
# TRAINING
###############################################################################
###############################################################################

# training information
encoder = EncoderCNN(EMBED_SIZE)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size)
loss_func = nn.CrossEntropyLoss()
learnable_params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params=learnable_params, lr=0.001)

train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)

print('training steps:', train_step)
print('validation steps:', val_step)

if torch.cuda.is_available():
    decoder = decoder.cuda()
    encoder = encoder.cuda()
    loss_func = nn.CrossEntropyLoss().cuda()

# losses
train_loss = []
val_loss = []
val_bleu_scores = [] # BLEU - Bilingual Evaluation Understudy score - widely adopted, inexpensive evaluation of sentence. 1 is perfect match, 0 is perfect mismatch
best_bleu_score = float('-INF')


for epoch in range(0, EPOCHS):
    if epoch == 0:
        train_checkpoint = torch.load('./models/train-model-018900.pkl')
        encoder.load_state_dict(train_checkpoint['encoder'])
        decoder.load_state_dict(train_checkpoint['decoder'])
        optimizer.load_state_dict(train_checkpoint['optimizer'])
        # epoch = train_checkpoint['epoch']

        start_loss = train_checkpoint['total_loss']
        start_step = train_checkpoint['train_step'] + 1
        start_time = time.time()
        train_loss = train(train_loader, encoder, decoder, loss_func, optimizer, vocab_size, epoch, train_step, start_step, start_loss)
    else:
        t_loss = train(train_loader, 
                    encoder, 
                    decoder, 
                    loss_func, 
                    optimizer, 
                    vocab_size, 
                    epoch, 
                    train_step)
    v_loss, bleu_score = validate(val_loader, 
                                  encoder, 
                                  decoder, 
                                  loss_func, 
                                  train_loader.dataset.vocab, 
                                  epoch, 
                                  val_step)
                                        
    train_loss.append(t_loss)
    val_loss.append(v_loss)
    val_bleu_scores.append(bleu_score)

    if bleu_score > best_bleu_score:
        print('Validation Bleu score improved from {:0.4f} to {:0.4f}, saving model to best_model.pkl'.format(best_bleu_score, bleu_score))
        best_bleu_score = bleu_score
        fname = os.path.join('./models', 'best_model.pkl')
        save_epoch(fname, encoder, decoder, optimizer, train_loss, val_loss, bleu_score, val_bleu_scores, epoch)
    else:
        print('saving model to model_{}.pkl'.format(epoch))
        fname = os.path.join('./models', 'model_{}.pkl'.format(epoch))
        save_epoch(fname, encoder, decoder, optimizer, train_loss, val_loss, bleu_score, val_bleu_scores, epoch)
        print('Epoch [%d/%d] took %ds' % (epoch, EPOCHS, time.time() - start_time))
    
    if epoch > 5:
        if early_stopping(val_bleu_scores, e):
            break
    start_time = time.time()






# features = encoder(images)


# if torch.cuda.is_available():
#     captions = captions.cuda()

# outputs = decoder(features, captions)