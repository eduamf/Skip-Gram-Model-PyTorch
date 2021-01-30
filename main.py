from __future__ import print_function
from tqdm import tqdm
# from tqdm import tqdm_gui
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, shutil, pickle
#from pprint import pprint 

import torch
import torch.optim as optim
# import torch.nn as nn

# it is a little tricky on run SummaryWriter by installing a suitable version of pytorch. so if you are able to import SummaryWriter from torch.utils.tensorboard, this script will record summaries. Otherwise it would not.
try:
    from torch.utils.tensorboard import SummaryWriter
    write_summary = True
except:
    write_summary = False

from model import Word2Vec_neg_sampling
from utils_modified import count_parameters
from datasets import word2vec_dataset
import config as cfg
# from test import print_nearest_words
# from utils_modified import q

# for tensorboard to work properly on embeddings projections
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# remove MODEL_DIR if it exists
if os.path.exists(cfg.MODEL_DIR):
    shutil.rmtree(cfg.MODEL_DIR)
# create MODEL_DIR    
os.makedirs(cfg.MODEL_DIR)

# SUMMARY_DIR is the path of the directory where the tensorboard SummaryWriter files are written
if write_summary:
    if os.path.exists(cfg.SUMMARY_DIR):
        # the directory is removed, if it already exists
        shutil.rmtree(cfg.SUMMARY_DIR)

    writer = SummaryWriter(cfg.SUMMARY_DIR) # this command automatically creates the directory at SUMMARY_DIR
    summary_counter = 0

# make training data
if not os.path.exists(cfg.PREPROCESSED_DATA_PATH):
    train_dataset = word2vec_dataset(cfg.DATA_SOURCE, cfg.CONTEXT_SIZE, cfg.FRACTION_DATA,
                                     cfg.SUBSAMPLING, cfg.SAMPLING_RATE)

    if not os.path.exists(cfg.PREPROCESSED_DATA_DIR):
        os.makedirs(cfg.PREPROCESSED_DATA_DIR)

    # pickle dump
    print('\ndumping pickle...')
    outfile = open(cfg.PREPROCESSED_DATA_PATH,'wb')
    pickle.dump(train_dataset, outfile)
    outfile.close()
    print('pickle dumped\n')

else:
    # pickle load
    print('\nloading pickle...')
    infile = open(cfg.PREPROCESSED_DATA_PATH,'rb')
    train_dataset = pickle.load(infile)
    infile.close()
    print('pickle loaded\n')

vocab = train_dataset.vocab
word_to_ix = train_dataset.word_to_ix
ix_to_word = train_dataset.ix_to_word

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.BATCH_SIZE,
                                           shuffle = not True)
print('len(train_dataset): ', len(train_dataset))
print('len(train_loader): ', len(train_loader))
print('len(vocab): ', len(vocab), '\n')

# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

losses = []

model = Word2Vec_neg_sampling(cfg.EMBEDDING_DIM, len(vocab), cfg.DEVICE, noise_dist,
                              cfg.NEGATIVE_SAMPLES).to(cfg.DEVICE)
print('\nWe have {} Million trainable parameters here in the model'.format(count_parameters(model)))

# optimizer = optim.SGD(model.parameters(), lr = 0.008, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr = cfg.LR)
# print(model, '\n')

for epoch in tqdm(range(cfg.NUM_EPOCHS)):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, cfg.NUM_EPOCHS))    
    # print('\nTRAINING...')

    # model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        print('batch# ' + str(batch_idx+1).zfill(len(str(len(train_loader)))) + '/' + str(len(train_loader)), end = '\r')
        
        model.train()

        x_batch           = x_batch.to(cfg.DEVICE)
        y_batch           = y_batch.to(cfg.DEVICE)
        
        optimizer.zero_grad()
        loss = model(x_batch, y_batch)
        
        loss.backward()
        optimizer.step()    
        
        losses.append(loss.item())
        if write_summary:
            # write tensorboard summaries
            writer.add_scalar(f'batch_loss', loss.item(), summary_counter)
            summary_counter += 1

        if batch_idx % cfg.DISPLAY_EVERY_N_BATCH == 0 and cfg.DISPLAY_BATCH_LOSS:
            print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')    
            # show 5 closest words to some test words
            # print_nearest_words(model, TEST_WORDS, word_to_ix, ix_to_word, top = 5)        

    # write embeddings every SAVE_EVERY_N_EPOCH epoch
    if epoch % cfg.SAVE_EVERY_N_EPOCH == 0:      
        writer.add_embedding(model.embeddings_input.weight.data,
                             metadata=[ix_to_word[k] for k in range(len(ix_to_word))],
                             global_step=epoch)

        torch.save({'model_state_dict': model.state_dict(), 
                    'losses': losses,
                    'word_to_ix': word_to_ix,
                    'ix_to_word': ix_to_word
                    },                  
                    '{}/model{}.pth'.format(cfg.MODEL_DIR, epoch))

plt.figure(figsize = (50, 50))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch")

plt.plot(losses)
plt.savefig('losses.png')
plt.show()

# '''
EMBEDDINGS = model.embeddings_input.weight.data
print('EMBEDDINGS.shape: ', EMBEDDINGS.shape)

from sklearn.manifold import TSNE

print('\n', 'running TSNE...')
tsne = TSNE(n_components = 2).fit_transform(EMBEDDINGS.cpu())
print('tsne.shape: ', tsne.shape) #(15, 2)

############ VISUALIZING ############
x, y = [], []
annotations = []
for idx, coord in enumerate(tsne):
    # print(coord)
    annotations.append(ix_to_word[idx])
    x.append(coord[0])
    y.append(coord[1])   

# test_words = ['king', 'queen', 'berlin', 'capital', 'germany', 'palace', 'stays']
# test_words = ['sun', 'moon', 'earth', 'while', 'open', 'run', 'distance', 'energy', 'coal', 'exploit']
# test_words = ['amazing', 'beautiful', 'work', 'breakfast', 'husband', 'hotel', 'quick', 'cockroach']

test_words = cfg.TEST_WORDS_VIZ
print('test_words: ', test_words)

plt.figure(figsize = (50, 50))
for i in range(len(test_words)):
    word = test_words[i]
    #print('word: ', word)
    vocab_idx = word_to_ix[word]
    # print('vocab_idx: ', vocab_idx)
    plt.scatter(x[vocab_idx], y[vocab_idx])
    plt.annotate(word, xy = (x[vocab_idx], y[vocab_idx]), \
        ha='right',va='bottom')

plt.savefig("w2v.png")
plt.show()
# '''