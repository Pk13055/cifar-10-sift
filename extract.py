#!/usr/bin/env python3
"""
    Script extracts the required classes
    into a separate dataset of 1000 images per class
    Labels are represented in the same way and key encoding as
    specified in the `batches.meta` file.
"""

from sys import argv as rd
import os
from multiprocessing import Pool, Process
import numpy as np
import random
import pickle

DATA_DIR = os.path.join(os.getcwd(), 'dataset')

def getImage(img_mat, plot=False):
    '''
        @description returns a 32x32 image given a single row
        repr of the image
        _Optionally plots the image_
        @param img_mat -> np.array: |img_mat| = (3072, ) OR (3072, 1)
        @param plot -> bool: whether to plot it or not
        @return image_repr: np.ndarray |image_repr| = (32, 32, 3)
    '''
    assert img_mat.shape in [(3072,), (3072, 1)] # sanity check
    r_channel = img_mat[:1024].reshape(32, 32)
    g_channel = img_mat[1024: 2 * 1024].reshape(32, 32)
    b_channel = img_mat[2 * 1024:].reshape(32, 32)
    image_repr = np.stack([r_channel, g_channel, b_channel], axis=2)
    assert image_repr.shape == (32, 32, 3) # sanity check
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(image_repr), plt.show(block=False)

    return image_repr

def getBatch(batchfile, category):
    '''
        @description parse a batchfile and extract the
        given category records
        @param batchfile -> str: the batch filename to parse
        @param category -> str: the category label records to extract
        @return category_mat -> np.array: Array of records matching the given
        category
    '''
    batch_data = pickle.load(open(os.path.join(DATA_DIR, batchfile), 'rb'), encoding='bytes')
    data = batch_data[bytes('data'.encode())]
    labels = np.array(batch_data[bytes('labels'.encode())])
    titles = pickle.load(open(os.path.join(DATA_DIR, 'batches.meta'),'rb'))['label_names']
    indices = np.where(labels == titles.index(category))
    category_mat = data[indices]
    random.shuffle(category_mat)
    # _ = getImage(random.choice(category_mat), plot=True)
    return category_mat


def getCategory(category, samples=None):
    '''
        @description Get the given category from the dataset
        @param category -> str: Category name
        @return category_obj -> np.array: object containing the records matching that category
            NOTE: There will a label attached with every image as well
    '''
    batch_pool = Pool(5)
    targets = [('data_batch_%d' % _, category) for _ in range(1, 6)]
    pool = batch_pool.starmap_async(getBatch, targets)
    batch_list = pool.get()
    print("Category %s batches : " % category, [_.shape for _ in batch_list])
    category_obj = np.vstack(batch_list)
    if samples:
        indices = random.sample(range(category_obj.shape[0]), samples)
        category_obj = category_obj[indices]
    print("Category %s size" % category, category_obj.shape)
    pickle.dump(category_obj, open('%s_pickled.pkl' % category, 'wb'))
    return category_obj


def main():
    labels = pickle.load(open(os.path.join(DATA_DIR, 'batches.meta'), 'rb'))
    print('Available categories', labels['label_names'])
    batch_size = int(rd[1] or 5000)
    categories = rd[2:]
    print('Categories chosen', categories)
    processes = [Process(target=getCategory, args=(_,batch_size)) for _ in categories]
    [_.start() for _ in processes]
    [_.join() for _ in processes]


if __name__ == "__main__":
    main()
