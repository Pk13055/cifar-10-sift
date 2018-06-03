#!/usr/bin/env python3
"""
    encode the current dataset into the bag of words
    feature vector
"""
import pickle
from multiprocessing import Pool
from sys import argv as rd

import cv2
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()
centroids = pickle.load(open('dataset/descriptors.pkl', 'rb'))


def getImage(img_mat, gray=True, plot=False):
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

    return cv2.cvtColor(image_repr, cv2.COLOR_RGB2GRAY)


def getSIFT(img):
    '''
        @description Get the SIFT features of the input image
        @param img -> np.array: |img| => { (32, 32), (3072, 1||0) }
        @return descriptor -> np.array n x 128
    '''
    if img.shape in [(3072, 1), (3072,)]: img = getImage(img)
    kps, des = sift.detectAndCompute(img, None)
    return des if des is not None else np.array([]).reshape(0, 128)


def get_inpv(descriptor):
    '''
        @description helper function to perfom async tasks
        calculates the actual nx1 vector for each image
        @param descriptor -> list: list of sift features for a given image
            |descriptor| = a * 128, variable a
        @return inp_vec -> np.array: bag of words repr for the given image
            |inp_vec| = (1, n), n => number of centroids
    '''
    counts = {}
    bag_size = centroids.shape[0]
    distances = [np.argmax(np.sqrt(np.sum((_ - centroids ) ** 2 , axis=1))) for _ in descriptor]
    [counts.update({_ : distances.count(_)}) for _ in range(bag_size)]
    inp_vec = np.zeros(bag_size)
    for _ in counts: inp_vec[_] = counts[_]
    return inp_vec


def normalizeInput(descriptors):
    '''
        @param descriptors -> list: each images descriptors.
        @param centroids -> np.array: n x 128
            Finally dimensionality of the input will be n x 1
        @return input_array -> np.array: n x 1
    '''
    pool = Pool(4)
    bag_size = centroids.shape[0]
    input_array = pool.map_async(get_inpv, descriptors).get()
    return np.array(input_array) / bag_size


def main():
    filename = rd[1]
    data = pickle.load(open(filename, 'rb'))
    pool = Pool(4)
    data_des = pool.map_async(getSIFT, data).get()
    data_inps = normalizeInput(data_des)
    print("Final dataset : ", data_inps.shape)
    pickle.dump(data_inps, open('BOW_repr_%s' % filename, 'wb'))

if __name__ == '__main__':
    main()
